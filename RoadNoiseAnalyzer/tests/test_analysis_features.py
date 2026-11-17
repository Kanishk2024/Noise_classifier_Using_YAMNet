import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.classifier import aggregate_scores
from src.noise_level import compute_noise_levels


class AnalysisFeatureTests(unittest.TestCase):
    def test_classifier_returns_five_requested_groups(self):
        names = [
            "Car",
            "Vehicle horn, car horn, honking",
            "Bird",
            "Speech",
            "Siren",
        ]
        scores = np.array([[0.2, 0.3, 0.4, 0.5, 0.9]], dtype=np.float32)

        result = aggregate_scores(scores, names, duration_sec=0.96)
        groups = {item["group"]: item for item in result["grouped_summary"]}

        self.assertEqual(
            set(groups),
            {"Vehicle", "Horn", "Nature", "Human", "Extra noise"},
        )
        self.assertEqual(groups["Extra noise"]["top_members"], ["Siren"])
        self.assertEqual(groups["Extra noise"]["mean_score"], 0.9)

    def test_noise_series_includes_short_final_audio_segment(self):
        sample_rate = 16000
        samples = np.sin(
            2 * np.pi * 440 * np.arange(int(sample_rate * 1.25)) / sample_rate
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "short-tail.wav"
            sf.write(wav_path, samples, sample_rate)

            result = compute_noise_levels(str(wav_path), calibration_db=100.0)

        self.assertEqual([point["t"] for point in result["time_series"]], [0, 1])

    def test_zero_calibration_offset_is_still_calibrated(self):
        sample_rate = 16000
        samples = np.sin(
            2 * np.pi * 440 * np.arange(sample_rate) / sample_rate
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "zero-offset.wav"
            sf.write(wav_path, samples, sample_rate)

            result = compute_noise_levels(str(wav_path), calibration_db=0.0)

        self.assertTrue(result["calibrated"])
        self.assertEqual(result["calibration_offset_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
