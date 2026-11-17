# RoadNoiseAnalyzer

Analyse a road recording (video or audio) and get:

1. **Noise levels in dBA** - LAeq, LAFmax, LAmin (plus L10/L50/L90)
2. A **per-second dBA time series** across the full recording (JSON + CSV)
3. **Five sound categories** from Google's YAMNet: Vehicle, Horn, Nature, Human, and Extra noise
4. A **report** as `report.json` + `report.txt`
5. A **plot** `noise_timeseries.png` (dBA over time, LAeq as dashed line)
6. A **Streamlit frontend** for uploading a recording and viewing its dB plot and categories

## Install

```bash
# 1. ffmpeg must be on your PATH
#    Windows: winget install ffmpeg   | macOS: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required. The first run downloads YAMNet (~15 MB) from TF Hub and the
class-map CSV (cached in `.cache/`), so it needs internet once.

## Usage

```bash
streamlit run app.py
python main.py samples/road.mp4
python main.py samples/road.wav --top-k 15 --out-dir output/run1
python main.py samples/road.mp4 --calibration-db 113.99          # known offset
python main.py samples/road.mp4 --calib-file calib.wav --calib-ref-db 94   # derive offset
python main.py samples/road.mp4 --no-plot --keep-wav
```

| Option | Meaning |
|---|---|
| `input_path` | video (.mp4 .mov .mkv .avi) or audio (.wav, .mp3, ...) |
| `--calibration-db` | offset in dB: `dBA = dBFS(A) + offset` (default: none -> relative values) |
| `--calib-file`, `--calib-ref-db` | derive the offset from a calibrator recording (default ref 94 dB) |
| `--out-dir` | output folder (default `output`) |
| `--top-k` | number of classes to report (default 10) |
| `--no-plot` | skip the PNG |
| `--keep-wav` | keep the extracted 16 kHz mono wav in the output folder |
| `--config` | alternate YAML config |

In the Streamlit app, upload an audio or video file and select **Analyze recording**.
The optional calibration offset is available in the sidebar. Without it, displayed
levels are relative.

Exit codes: 0 ok, 1 analysis error, 2 input not found, 3 ffmpeg missing.

## Output files (in `--out-dir`)

`report.json`, `report.txt`, `noise_timeseries.csv`, `noise_timeseries.png` (unless `--no-plot`).

## How it works

- `src/audio_extract.py` - `ffmpeg -i in -ac 1 -ar 16000 -sample_fmt s16 out.wav -y`
- `src/noise_level.py` - IEC A-weighting filter (SOS), Fast (125 ms) time weighting for LAFmax/LAmin, energy average for LAeq, 1-second windows for the series
- `src/classifier.py` - YAMNet frames (0.48 s hop) averaged into 1-second windows; sound classes are summarized as Vehicle, Horn, Nature, Human, or Extra noise
- `src/report.py` - JSON/text/plot output
- `app.py` - Streamlit upload, analysis, dB chart, and category summary

## Important notes

- **Calibration matters.** Without it the dBA values are *relative*; the report and plot say so. See `calibration/README.md`.
- **`soundlevelmeter` deviation.** The PyPI package of that name installs a module called `slm` with a different API than the one in the original spec (and 1.0+ needs Python 3.13), so the dBA maths is implemented directly in `noise_level.py` using SciPy. The A-weighting curve was checked against IEC values (within ~0.1 dB up to 2 kHz, ~0.5 dB at 4 kHz at 16 kHz sampling).
- Results from phone/camera audio are indicative only (auto-gain, wind noise, clipping); this is not a compliance-grade measurement.
- YAMNet scores are per-class sigmoid outputs, not probabilities that sum to 1; several classes (Vehicle, Car, Engine) can be high at once.

## Layout

```
RoadNoiseAnalyzer/
├── main.py  config.yaml  requirements.txt  README.md
├── src/ (audio_extract, noise_level, classifier, report, utils)
├── calibration/README.md
├── samples/   <- put your files here
└── output/    <- reports land here
```
