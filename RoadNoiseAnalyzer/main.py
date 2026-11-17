#!/usr/bin/env python3
"""RoadNoiseAnalyzer CLI: dBA levels + sound categories from an audio/video file."""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import tempfile
from pathlib import Path

from src import __version__
from src.audio_extract import FFmpegNotFoundError, extract_audio
from src.classifier import classify_audio
from src.noise_level import calibrate_from_file, compute_noise_levels
from src.report import build_report, render_text_report, save_noise_plot, save_report
from src.utils import ensure_dir, get_logger, load_config, setup_logging

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse road noise (dBA) and sound categories.")
    p.add_argument("input_path", help="video (.mp4/.mov/.mkv/.avi) or audio (.wav/...) file")
    p.add_argument("--calibration-db", type=float, default=None,
                   help="calibration offset in dB (dBA = dBFS(A) + offset); default: config.yaml / none")
    p.add_argument("--calib-file", default=None,
                   help="recording of a calibrator tone; the offset is derived from it")
    p.add_argument("--calib-ref-db", type=float, default=94.0,
                   help="level of the calibrator in dB SPL (default 94.0)")
    p.add_argument("--out-dir", default="output", help="where reports land (default: output)")
    p.add_argument("--top-k", type=int, default=10, help="top-K sound classes (default 10)")
    p.add_argument("--no-plot", action="store_true", help="skip the PNG plot")
    p.add_argument("--keep-wav", action="store_true", help="keep the extracted 16 kHz wav")
    p.add_argument("--config", default=None, help="path to an alternate config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"RoadNoiseAnalyzer {__version__}")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    # 1. validate input
    in_path = Path(args.input_path)
    if not in_path.is_file():
        log.error("Input file not found: %s", in_path)
        return 2

    out_dir = ensure_dir(args.out_dir)
    keep_wav = args.keep_wav or cfg["output"]["keep_wav"]
    make_plot = not args.no_plot and cfg["output"]["plot"]

    # calibration priority: calibrator recording > CLI offset > config
    calibration_db = cfg["noise"]["calibration_db"]
    if args.calibration_db is not None:
        calibration_db = args.calibration_db
    if args.calib_file:
        calibration_db = calibrate_from_file(args.calib_file, args.calib_ref_db)

    tmp_dir = Path(tempfile.mkdtemp(prefix="roadnoise_"))
    wav_path = tmp_dir / f"{in_path.stem}_16k.wav"
    try:
        # 2. extract audio
        extract_audio(str(in_path), str(wav_path))

        # 3. noise levels
        noise = compute_noise_levels(str(wav_path), calibration_db)

        # 4. classification
        c = cfg["classification"]
        classes = classify_audio(
            str(wav_path), top_k=args.top_k, window_sec=c["window_sec"],
            threshold=c["threshold"], model_url=c["model_url"], class_map_url=c["class_map_url"],
        )

        # 5. report
        meta = {
            "input_file": str(in_path),
            "analyzed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tool_version": __version__,
            "top_k": args.top_k,
            "threshold": c["threshold"],
        }
        report = build_report(noise, classes, meta)
        save_report(report, str(out_dir))

        # 6. plot
        if make_plot:
            log.info("Saved plot %s", save_noise_plot(noise, str(out_dir)))

        # 7. print report
        print(render_text_report(report))

        if keep_wav:
            kept = out_dir / wav_path.name
            shutil.copy2(wav_path, kept)
            log.info("Kept wav: %s", kept)
        return 0

    except FFmpegNotFoundError as exc:
        log.error("%s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001 - CLI top-level
        log.error("Analysis failed: %s", exc, exc_info=args.verbose)
        return 1
    finally:
        # 8. clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
