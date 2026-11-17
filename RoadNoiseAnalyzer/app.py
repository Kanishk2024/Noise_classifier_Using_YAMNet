"""Streamlit frontend for road-noise analysis."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audio_extract import extract_audio
from src.classifier import classify_audio
from src.noise_level import compute_noise_levels
from src.utils import load_config

st.set_page_config(page_title="Road Noise Analyzer", page_icon="🔊", layout="wide")
st.title("Road Noise Analyzer")
st.caption("Upload a recording to view its sound categories and dB level over time.")

config = load_config()
classification_config = config["classification"]
configured_calibration = config["noise"]["calibration_db"]

uploaded_file = st.file_uploader(
    "Audio or video file",
    type=["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mov", "mkv", "avi", "webm"],
)

with st.sidebar:
    st.header("Measurement")
    use_calibration = st.checkbox(
        "Use calibration offset",
        value=configured_calibration is not None,
        help="Without calibration, the displayed dB values are relative, not absolute dBA.",
    )
    calibration_offset = st.number_input(
        "Calibration offset (dB)",
        value=float(configured_calibration or 0.0),
        step=0.1,
        disabled=not use_calibration,
    )

upload_bytes = uploaded_file.getvalue() if uploaded_file else None
upload_digest = hashlib.sha256(upload_bytes).hexdigest() if upload_bytes is not None else None

if uploaded_file and st.button("Analyze recording", type="primary"):
    st.session_state.pop("analysis_result", None)
    with st.spinner("Analyzing audio and detecting sound categories..."):
        try:
            with tempfile.TemporaryDirectory(prefix="roadnoise_") as temp_dir:
                source_path = Path(temp_dir) / f"input{Path(uploaded_file.name).suffix.lower()}"
                wav_path = Path(temp_dir) / "analysis.wav"
                source_path.write_bytes(upload_bytes)
                extract_audio(str(source_path), str(wav_path))

                noise = compute_noise_levels(
                    str(wav_path),
                    float(calibration_offset) if use_calibration else None,
                )
                categories = classify_audio(
                    str(wav_path),
                    window_sec=classification_config["window_sec"],
                    threshold=classification_config["threshold"],
                    model_url=classification_config["model_url"],
                    class_map_url=classification_config["class_map_url"],
                )

            st.session_state["analysis_result"] = {
                "upload_digest": upload_digest,
                "file_name": uploaded_file.name,
                "noise": noise,
                "categories": categories["grouped_summary"],
            }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis_result")
if result and result["upload_digest"] == upload_digest:
    noise = result["noise"]
    unit = "dBA" if noise["calibrated"] else "dB (relative)"
    if not noise["calibrated"]:
        st.warning(
            "No calibration offset was applied. Levels are relative and should not be "
            "interpreted as absolute sound pressure (dBA)."
        )

    st.subheader(result["file_name"])
    metric_columns = st.columns(3)
    metric_columns[0].metric(f"LAeq ({unit})", noise["laeq_dba"])
    metric_columns[1].metric(f"Maximum ({unit})", noise["lafmax_dba"])
    metric_columns[2].metric("Duration", f'{noise["duration_sec"]:.1f} s')

    st.subheader(f"{unit} throughout the recording")
    time_series = pd.DataFrame(noise["time_series"]).rename(
        columns={"t": "Time (s)", "dba": unit}
    )
    st.line_chart(time_series, x="Time (s)", y=unit, height=360)

    st.subheader("Sound categories")
    category_table = pd.DataFrame(result["categories"]).rename(
        columns={
            "group": "Category",
            "mean_score": "Average score",
            "max_score": "Peak score",
            "time_pct": "Time detected",
            "top_members": "Top sounds",
        }
    )
    for column in ("Average score", "Peak score", "Time detected"):
        category_table[column] = (category_table[column] * 100).round(1)
    st.dataframe(category_table, hide_index=True, use_container_width=True)
elif result and uploaded_file is None:
    st.info("Upload the analyzed file again to view its results.")
