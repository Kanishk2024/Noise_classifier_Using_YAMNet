"""Build and save JSON + human-readable reports, and the dBA time-series plot."""
from __future__ import annotations

import json
from pathlib import Path

from .utils import ensure_dir, get_logger

log = get_logger(__name__)

# Indicative wording only (not a regulatory classification); needs calibrated data.
_LEVEL_BANDS = [(50, "Quiet"), (60, "Moderate"), (70, "Loud"), (80, "Very loud"), (999, "Extremely loud")]


def _describe_level(laeq: float, calibrated: bool) -> str:
    if not calibrated:
        return "n/a (uncalibrated - values are relative)"
    for upper, label in _LEVEL_BANDS:
        if laeq < upper:
            return label
    return "n/a"


def build_report(noise_data: dict, class_data: dict, meta: dict) -> dict:
    """Combine noise + classification results into one JSON-serialisable dict."""
    calibrated = bool(noise_data.get("calibrated"))
    summary_keys = (
        "laeq_dba", "lafmax_dba", "lamin_dba", "l10_dba", "l50_dba", "l90_dba",
        "duration_sec", "calibrated", "calibration_offset_db",
    )
    return {
        "meta": meta,
        "noise_summary": {k: noise_data[k] for k in summary_keys if k in noise_data}
        | {"indicative_level": _describe_level(noise_data["laeq_dba"], calibrated)},
        "time_series": noise_data["time_series"],
        "sound_categories": class_data["top_classes"],
        "grouped_summary": class_data["grouped_summary"],
        "per_window": class_data["per_window"],
        "warnings": (
            [] if calibrated else
            ["Uncalibrated: dBA values are RELATIVE, not absolute sound pressure levels."]
        ),
    }


def render_text_report(report: dict) -> str:
    m, n = report["meta"], report["noise_summary"]
    unit = "dBA" if n["calibrated"] else "dBA (relative)"
    L = []
    L.append("=" * 62)
    L.append("  ROAD NOISE ANALYSIS REPORT")
    L.append("=" * 62)
    L.append(f"File       : {m.get('input_file', '?')}")
    L.append(f"Analysed   : {m.get('analyzed_at', '?')}")
    L.append(f"Duration   : {n['duration_sec']:.1f} s")
    L.append(f"Calibrated : {'yes (offset %+.2f dB)' % n['calibration_offset_db'] if n['calibrated'] else 'NO - relative values only'}")
    L.append("")
    L.append("NOISE LEVELS")
    L.append("-" * 62)
    L.append(f"  LAeq   (average)        : {n['laeq_dba']:7.1f} {unit}")
    L.append(f"  LAFmax (loudest, Fast)  : {n['lafmax_dba']:7.1f} {unit}")
    L.append(f"  LAmin  (quietest, Fast) : {n['lamin_dba']:7.1f} {unit}")
    L.append(f"  L10 / L50 / L90         : {n['l10_dba']:.1f} / {n['l50_dba']:.1f} / {n['l90_dba']:.1f}")
    L.append(f"  Indicative level        : {n['indicative_level']}")
    L.append("")
    L.append("SOUND GROUPS (mean score, % of time detected)")
    L.append("-" * 62)
    for g in report["grouped_summary"][:6]:
        L.append(f"  {g['group']:<16} {g['mean_score'] * 100:5.1f}%   present {g['time_pct'] * 100:5.1f}% of time")
    L.append("")
    L.append("TOP SOUND CATEGORIES")
    L.append("-" * 62)
    for c in report["sound_categories"]:
        L.append(f"  {c['name'][:34]:<34} {c['mean_score'] * 100:5.1f}%  (max {c['max_score'] * 100:3.0f}%, {c['time_pct'] * 100:3.0f}% of time)")
    if report.get("warnings"):
        L.append("")
        for w in report["warnings"]:
            L.append(f"! {w}")
    L.append("=" * 62)
    return "\n".join(L)


def save_report(report: dict, out_dir: str) -> tuple[str, str]:
    """Write report.json and report.txt into `out_dir`; return their paths."""
    out = ensure_dir(out_dir)
    json_path, txt_path = out / "report.json", out / "report.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(render_text_report(report) + "\n", encoding="utf-8")
    # bonus: the per-second dBA series as CSV (handy for Excel / Power BI)
    import pandas as pd
    pd.DataFrame(report["time_series"]).rename(columns={"t": "time_sec"}).to_csv(
        out / "noise_timeseries.csv", index=False)
    log.info("Saved %s and %s", json_path, txt_path)
    return str(json_path), str(txt_path)


def save_noise_plot(noise_data: dict, out_dir: str) -> str:
    """Plot per-second dBA with LAeq as a dashed horizontal line."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    ts = noise_data["time_series"]
    t = [p["t"] for p in ts]
    y = [p["dba"] for p in ts]
    calibrated = noise_data.get("calibrated")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, lw=1.6, label="Level (1 s)")
    ax.axhline(noise_data["laeq_dba"], ls="--", c="crimson", label=f"LAeq = {noise_data['laeq_dba']:.1f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dBA" if calibrated else "dBA (relative)")
    ax.set_title("Noise level over time" + ("" if calibrated else " - UNCALIBRATED"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(ensure_dir(out_dir)) / "noise_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
