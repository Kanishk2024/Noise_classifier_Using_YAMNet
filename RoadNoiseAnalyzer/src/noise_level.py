"""dBA noise levels (LAeq, LAFmax, LAmin) and a per-second dBA time series.

Implementation note
-------------------
The spec asked for the PyPI `soundlevelmeter` package, but that package installs a module
called `slm` with a different API (no `SoundLevelMeter.from_file`). To keep this project
dependable, the sound-level maths is implemented here directly:

  * A-weighting: analogue IEC 61672 A-weighting poles/zeros, bilinear-transformed to the
    file's sample rate (second-order sections).
  * LAeq: energy average of the A-weighted signal over the whole file.
  * LAFmax / LAmin: max / min of the A-weighted level with 'Fast' (125 ms) exponential
    time weighting, as on a hand-held sound level meter.
  * Time series: A-weighted mean-square per 1-second window -> dB.

Accuracy caveats: at 16 kHz sampling the bilinear transform deviates from the ideal curve
above ~4 kHz, and the result is only as good as the microphone/phone that recorded it.
Absolute dBA is only meaningful with a calibration offset (see calibration/README.md).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

from .utils import get_logger

log = get_logger(__name__)

EPS = 1e-12
FAST_TAU = 0.125  # seconds, IEC 'Fast' time weighting
# Used only when no calibration is supplied: treats 0 dBFS as 94 dB so numbers look like
# dB values. They are RELATIVE and must not be quoted as real SPL.
UNCALIBRATED_NOMINAL_OFFSET_DB = 94.0


def a_weighting_sos(fs: float) -> np.ndarray:
    """Digital A-weighting filter (second-order sections) for sample rate `fs`."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997  # dB, normalises gain to 0 dB at 1 kHz
    w = lambda f: 2 * np.pi * f  # noqa: E731
    zeros = [0.0, 0.0, 0.0, 0.0]
    poles = [-w(f1)] * 2 + [-w(f4)] * 2 + [-w(f2), -w(f3)]
    gain = (w(f4) ** 2) * 10 ** (a1000 / 20)
    zd, pd, kd = signal.bilinear_zpk(zeros, poles, gain, fs)
    return signal.zpk2sos(zd, pd, kd)


def _load_mono(wav_path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(wav_path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def _a_weighted(x: np.ndarray, sr: int) -> np.ndarray:
    return signal.sosfilt(a_weighting_sos(sr), x)


def _db(mean_square, offset: float):
    return 10.0 * np.log10(np.maximum(mean_square, EPS)) + offset


def calibrate_from_file(calib_wav: str, reference_db: float = 94.0) -> float:
    """Derive the calibration offset from a recording of a calibrator tone.

    offset = reference_db - (A-weighted level of the recording in dBFS)
    Record only the steady calibrator tone (trim any handling noise).
    """
    x, sr = _load_mono(calib_wav)
    xa = _a_weighted(x, sr)
    xa = xa[int(0.25 * sr):] if len(xa) > sr else xa  # drop filter settling
    level_dbfs = float(_db(np.mean(xa ** 2), 0.0))
    offset = reference_db - level_dbfs
    log.info("Calibrator recorded at %.2f dBFS -> offset %.2f dB", level_dbfs, offset)
    return offset


def compute_noise_levels(wav_path: str, calibration_db: float | None = None) -> dict:
    """Compute LAeq, LAFmax, LAmin and a per-second dBA series for `wav_path`.

    `calibration_db` is an offset in dB added to the A-weighted dBFS level
    (dBA = dBFS(A) + calibration_db). None -> uncalibrated, RELATIVE values.
    """
    calibrated = calibration_db is not None
    if not calibrated:
        log.warning(
            "No calibration_db given: dBA values are RELATIVE (nominal 0 dBFS = %.0f dB), "
            "NOT absolute sound pressure levels. See calibration/README.md.",
            UNCALIBRATED_NOMINAL_OFFSET_DB,
        )
        offset = UNCALIBRATED_NOMINAL_OFFSET_DB
    else:
        offset = float(calibration_db)

    x, sr = _load_mono(wav_path)
    if x.size == 0:
        raise ValueError(f"Audio file is empty: {wav_path}")
    duration = x.size / sr
    xa = _a_weighted(x, sr)
    p2 = xa ** 2

    # --- LAeq over the whole file
    laeq = float(_db(np.mean(p2), offset))

    # --- Fast-weighted level (125 ms exponential average) for LAFmax / LAmin
    alpha = np.exp(-1.0 / (FAST_TAU * sr))
    b, a = [1.0 - alpha], [1.0, -alpha]
    init = float(np.mean(p2[: max(1, int(FAST_TAU * sr))]))
    zi = signal.lfilter_zi(b, a) * init
    laf_ms, _ = signal.lfilter(b, a, p2, zi=zi)
    laf = _db(laf_ms, offset)
    lafmax = float(np.max(laf))
    lamin = float(np.min(laf))

    # --- per-second A-weighted levels
    win = int(sr)
    series = []
    for i, start in enumerate(range(0, x.size, win)):
        seg = p2[start:start + win]
        series.append({"t": i, "dba": round(float(_db(np.mean(seg), offset)), 2)})

    levels = np.array([s["dba"] for s in series])
    return {
        "laeq_dba": round(laeq, 2),
        "lafmax_dba": round(lafmax, 2),
        "lamin_dba": round(lamin, 2),
        # statistical levels (exceeded 10/50/90 % of the time, from 1-s levels)
        "l10_dba": round(float(np.percentile(levels, 90)), 2),
        "l50_dba": round(float(np.percentile(levels, 50)), 2),
        "l90_dba": round(float(np.percentile(levels, 10)), 2),
        "duration_sec": round(duration, 2),
        "sample_rate": int(sr),
        "calibrated": calibrated,
        "calibration_offset_db": round(offset, 2),
        "time_series": series,
    }
