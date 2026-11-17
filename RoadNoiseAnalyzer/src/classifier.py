"""YAMNet wrapper: per-window sound-category scores + file-level aggregation."""
from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

import numpy as np

from .utils import CACHE_DIR, get_logger

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow

log = get_logger(__name__)

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
SAMPLE_RATE = 16000
YAMNET_HOP_SEC = 0.48     # one score frame every 0.48 s
YAMNET_PATCH_SEC = 0.96   # each frame looks at 0.96 s of audio

# Focused groups -> exact AudioSet display names (as in yamnet_class_map.csv).
# Any class not listed here is included in "Extra noise".
CLASS_GROUPS: dict[str, list[str]] = {
    "Vehicle": [
        "Vehicle", "Motor vehicle (road)", "Car", "Car passing by", "Truck", "Bus",
        "Motorcycle", "Traffic noise, roadway noise", "Bicycle", "Train",
        "Railroad car, train wagon", "Aircraft", "Skidding", "Tire squeal",
    ],
    "Horn": [
        "Vehicle horn, car horn, honking", "Honking", "Car alarm", "Toot",
        "Air horn, truck horn", "Train horn",
    ],
    "Nature": [
        "Wind", "Wind noise (microphone)", "Rain", "Rain on surface", "Thunderstorm",
        "Thunder", "Bird", "Bird vocalization, bird call, bird song", "Dog", "Bark",
    ],
    "Human": [
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Shout", "Yell", "Crowd",
        "Laughter", "Whistling", "Cheering",
    ],
}

_MODEL = None
_CLASS_NAMES: list[str] | None = None


# --------------------------------------------------------------------------- loading
def _load_model(model_url: str = MODEL_URL):
    """Load YAMNet from TF Hub once and cache the handle."""
    global _MODEL
    if _MODEL is None:
        import tensorflow_hub as hub  # imported lazily: slow, and not needed for dBA only

        log.info("Loading YAMNet from %s (first run downloads ~15 MB)...", model_url)
        _MODEL = hub.load(model_url)
    return _MODEL


def _read_class_map(path: str | Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [row["display_name"] for row in csv.DictReader(fh)]


def _load_class_names(class_map_url: str = CLASS_MAP_URL, model=None) -> list[str]:
    """Return the 521 AudioSet display names (download the CSV if not cached)."""
    global _CLASS_NAMES
    if _CLASS_NAMES is not None:
        return _CLASS_NAMES

    csv_path = CACHE_DIR / "yamnet_class_map.csv"
    if not csv_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            log.info("Downloading class map to %s", csv_path)
            urllib.request.urlretrieve(class_map_url, csv_path)
        except Exception as exc:  # network problem -> use the copy bundled with the model
            log.warning("Class-map download failed (%s); using the model's bundled copy.", exc)
            if model is None:
                raise
            bundled = model.class_map_path().numpy().decode("utf-8")
            csv_path = Path(bundled)

    _CLASS_NAMES = _read_class_map(csv_path)
    return _CLASS_NAMES


# ------------------------------------------------------------------------ aggregation
def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def aggregate_scores(
    frame_scores: np.ndarray,
    class_names: list[str],
    duration_sec: float,
    top_k: int = 10,
    window_sec: float = 1.0,
    threshold: float = 0.15,
) -> dict:
    """Turn raw YAMNet frame scores (N, 521) into windowed + file-level summaries.

    Frame i spans [0.48*i, 0.48*i + 0.96] s; it is assigned to the window containing its
    centre, so ~2 frames fall in each 1-second window and are averaged.
    """
    n_frames, n_classes = frame_scores.shape
    centres = np.arange(n_frames) * YAMNET_HOP_SEC + YAMNET_PATCH_SEC / 2
    n_windows = max(1, int(np.ceil(duration_sec / window_sec - 1e-9)))

    win_scores = np.zeros((n_windows, n_classes), dtype=np.float64)
    for w in range(n_windows):
        lo, hi = w * window_sec, (w + 1) * window_sec
        idx = np.where((centres >= lo) & (centres < hi))[0]
        if idx.size == 0:  # very short windows / file edges: use the nearest frame
            idx = np.array([int(np.argmin(np.abs(centres - (lo + hi) / 2)))])
        win_scores[w] = frame_scores[idx].mean(axis=0)

    # ---- per-window top-k above threshold
    per_window = []
    for w in range(n_windows):
        order = np.argsort(win_scores[w])[::-1][:top_k]
        top = [
            {"name": class_names[i], "score": _r(win_scores[w, i])}
            for i in order if win_scores[w, i] >= threshold
        ]
        per_window.append({"t": _r(w * window_sec, 2), "top": top})

    # ---- file-level per-class stats
    mean_s = win_scores.mean(axis=0)
    max_s = win_scores.max(axis=0)
    time_pct = (win_scores >= threshold).mean(axis=0)
    order = np.argsort(mean_s)[::-1][:top_k]
    top_classes = [
        {
            "name": class_names[i],
            "mean_score": _r(mean_s[i]),
            "max_score": _r(max_s[i]),
            "time_pct": _r(time_pct[i]),
        }
        for i in order
    ]

    # ---- grouped summary (group score per window = max over member classes)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    grouped = []
    grouped_idxs: set[int] = set()
    for group, members in CLASS_GROUPS.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        if not idxs:
            grouped.append(
                {
                    "group": group,
                    "mean_score": 0.0,
                    "max_score": 0.0,
                    "time_pct": 0.0,
                    "top_members": [],
                }
            )
            continue
        grouped_idxs.update(idxs)
        g = win_scores[:, idxs].max(axis=1)
        best = sorted(idxs, key=lambda i: mean_s[i], reverse=True)[:3]
        grouped.append(
            {
                "group": group,
                "mean_score": _r(g.mean()),
                "max_score": _r(g.max()),
                "time_pct": _r((g >= threshold).mean()),
                "top_members": [class_names[i] for i in best if mean_s[i] > 0.01],
            }
        )
    extra_idxs = [i for i in range(n_classes) if i not in grouped_idxs]
    extra_scores = (
        win_scores[:, extra_idxs].max(axis=1)
        if extra_idxs else np.zeros(n_windows, dtype=np.float64)
    )
    best_extra = sorted(extra_idxs, key=lambda i: mean_s[i], reverse=True)[:3]
    grouped.append(
        {
            "group": "Extra noise",
            "mean_score": _r(extra_scores.mean()),
            "max_score": _r(extra_scores.max()),
            "time_pct": _r((extra_scores >= threshold).mean()),
            "top_members": [
                class_names[i] for i in best_extra if mean_s[i] > 0.01
            ],
        }
    )
    grouped.sort(key=lambda d: d["mean_score"], reverse=True)

    return {
        "top_classes": top_classes,
        "grouped_summary": grouped,
        "per_window": per_window,
        "n_windows": n_windows,
        "window_sec": window_sec,
        "threshold": threshold,
    }


# ----------------------------------------------------------------------------- public
def classify_audio(
    wav_path: str,
    top_k: int = 10,
    window_sec: float = 1.0,
    threshold: float = 0.15,
    model_url: str = MODEL_URL,
    class_map_url: str = CLASS_MAP_URL,
) -> dict:
    """Run YAMNet on `wav_path` and return class statistics (see aggregate_scores)."""
    import librosa

    waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {wav_path}")
    duration = waveform.size / SAMPLE_RATE

    # YAMNet needs at least one 0.96 s patch
    padded = waveform
    if waveform.size < int(YAMNET_PATCH_SEC * SAMPLE_RATE):
        padded = np.pad(waveform, (0, int(YAMNET_PATCH_SEC * SAMPLE_RATE) - waveform.size))

    model = _load_model(model_url)
    class_names = _load_class_names(class_map_url, model)

    scores, _embeddings, _spectrogram = model(padded.astype(np.float32))
    scores = scores.numpy()  # (N_frames, 521)
    log.info("YAMNet produced %d frames for %.1f s of audio", scores.shape[0], duration)

    return aggregate_scores(scores, class_names, duration, top_k, window_sec, threshold)
