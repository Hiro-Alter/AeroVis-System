from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests_future.evaluate_captures import classification, group_rows, load_samples, mean_std, process_image, summarize


def row(filename, truth, prediction, pair_id=None, occlusion="none"):
    valid = prediction is not None
    return dict(filename=filename, gesture=truth, prediction=prediction, roi_valid=valid,
                hand_detected=valid, pair_id=pair_id, occlusion=occlusion,
                background="busy", lighting="lit", image_sha256=filename,
                roi_sha256=filename if valid else None, roi_box=[0, 0, 5, 5] if valid else None,
                inference_ms=2.0 if valid else None, processing_ms=5.0, model="resnet18")


class EvaluationTests(unittest.TestCase):
    def test_metrics_keep_errors_and_exclude_only_missing_roi(self):
        rows = [row("1", "a", "a"), row("2", "a", "b"), row("3", "b", "b"), row("4", "b", None)]
        result = classification(rows, ["a", "b"])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["f1_macro"], 2/3)
        self.assertEqual(result["confusion_counts"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(classification(rows, ["a", "b", "c"])["f1_macro"], 4/9)
        self.assertIsNone(classification([rows[-1]], ["a", "b"])["accuracy"])

    def test_occlusion_comparison_uses_complete_detectable_pairs(self):
        rows = [row("1", "a", "a", "pair1"), row("2", "a", "b", "pair1", "partial"),
                row("3", "a", "a", "pair2"), row("4", "a", None, "pair2", "partial")]
        groups = group_rows(rows)
        self.assertEqual(len(groups["partial_all"]), 2)
        self.assertEqual([r["filename"] for r in groups["paired_base_common_roi"]], ["1"])
        self.assertEqual([r["filename"] for r in groups["partial_common_roi"]], ["2"])

    def test_timing_spread_and_missing_inference(self):
        self.assertEqual(mean_std([]), dict(n=0, mean_ms=None, std_ms=None))
        self.assertEqual(mean_std([4]), dict(n=1, mean_ms=4, std_ms=None))
        self.assertEqual(mean_std([1, 2, 3]), dict(n=3, mean_ms=2, std_ms=1))
        base = [row("1", "a", "a"), row("2", "b", None)]
        result = summarize(base + [{**r, "model": "mobilenetv3_small"} for r in base], ["a", "b"])
        self.assertEqual(result["detection"][0]["sensitivity"], 0.5)
        timing = [r for r in result["timing"] if r["model"] == "resnet18" and r["group"] == "base_all"]
        self.assertEqual([r["n"] for r in timing], [1, 1, 1])
        self.assertEqual([r["mean_ms"] for r in timing], [2, 5, 5])

    def test_different_roi_between_models_is_rejected(self):
        first = row("1", "a", "a")
        second = {**first, "model": "mobilenetv3_small", "roi_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "ROI/deteccion diferente"):
            summarize([first, second], ["a", "b"])

    def test_processing_mirrors_like_camera_and_does_not_infer_without_roi(self):
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        vision, predictor = Mock(), Mock()
        vision.process.return_value = SimpleNamespace(roi=None)
        with patch("tests_future.evaluate_captures.time.perf_counter", side_effect=[1, 1.02]):
            result, prediction, elapsed = process_image(frame, vision, predictor, True)
        self.assertTrue(np.array_equal(vision.process.call_args.args[0], frame[:, ::-1]))
        self.assertIsNone(prediction)
        predictor.predict_roi.assert_not_called()
        self.assertAlmostEqual(elapsed, 20)
        roi = frame.copy()
        vision.process.return_value = SimpleNamespace(roi=SimpleNamespace(image_bgr=roi))
        process_image(frame, vision, predictor, False)
        self.assertIs(predictor.predict_roi.call_args.args[0], roi)

    def test_dataset_validation_preserves_sources_and_rejects_missing_pair_or_bad_path(self):
        temp = tempfile.TemporaryDirectory(prefix="_evaluation_", dir=ROOT / "tests")
        root = Path(temp.name)
        try:
            labels = [f"gesture{i}" for i in range(10)]
            samples = {}
            for hand in ("right", "left"):
                (root / hand).mkdir()
                for background in ("simple", "busy"):
                    for lighting in ("lit", "dim"):
                        for gesture in labels:
                            for repetition in (1, 2, 3):
                                stem = f"{hand}_{background}_{lighting}_{gesture}_{repetition:02d}"
                                pair = background == "busy" and lighting == "lit" and repetition < 3
                                for occlusion in (["none", "partial"] if pair else ["none"]):
                                    name = f"{stem}_{occlusion}.png"
                                    samples[name] = dict(file=f"{hand}/{name}", hand=hand, background=background,
                                                         lighting=lighting, gesture=gesture, repetition=repetition,
                                                         occlusion=occlusion, pair_id=stem if pair else None)
                                    self.assertTrue(cv2.imwrite(str(root / hand / name), np.zeros((8, 8, 3), np.uint8)))
            manifest = root / "annotations.json"
            original = json.dumps({"samples": samples})
            manifest.write_text(original, encoding="utf-8")
            self.assertEqual(len(load_samples(root, labels)), 280)
            self.assertEqual(manifest.read_text(encoding="utf-8"), original)
            paired = next(k for k, v in samples.items() if v["occlusion"] == "partial")
            samples.pop(paired)
            manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_samples(root, labels)
            samples = json.loads(original)["samples"]
            next(iter(samples.values()))["file"] = "../outside.png"
            manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_samples(root, labels)
        finally:
            self.assertEqual(root.resolve().parent, (ROOT / "tests").resolve())
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
