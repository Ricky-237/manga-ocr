from __future__ import annotations

import unittest

from manga_ocr.domain import DatasetSplit, Domain, PolygonPoint, TextDirection, TextType
from manga_ocr.eval.ocr_eval import GoldLineRecord, evaluate_predictions
from manga_ocr.pseudo_label import TeacherPrediction


class OcrEvalTests(unittest.TestCase):
    def test_evaluate_predictions_computes_overall_and_slice_metrics(self) -> None:
        gold_records = [
            GoldLineRecord(
                image_path="img-1.jpg",
                series_id="series-a",
                domain=Domain.MANGA,
                polygon=[
                    PolygonPoint(0, 0),
                    PolygonPoint(10, 0),
                    PolygonPoint(10, 10),
                    PolygonPoint(0, 10),
                ],
                transcript="hello world",
                lang="latin",
                direction=TextDirection.HORIZONTAL,
                text_type=TextType.DIALOGUE,
                split=DatasetSplit.TRAIN,
            ),
            GoldLineRecord(
                image_path="img-2.jpg",
                series_id="series-b",
                domain=Domain.WEBTOON,
                polygon=[
                    PolygonPoint(0, 0),
                    PolygonPoint(12, 0),
                    PolygonPoint(12, 12),
                    PolygonPoint(0, 12),
                ],
                transcript="bonjour",
                lang="latin",
                direction=TextDirection.VERTICAL,
                text_type=TextType.CAPTION,
                split=DatasetSplit.TEST,
            ),
        ]
        predictions = [
            TeacherPrediction(
                asset_id="asset-1",
                series_id="series-a",
                image_path="img-1.jpg",
                domain=Domain.MANGA,
                polygon=[
                    PolygonPoint(0, 0),
                    PolygonPoint(10, 0),
                    PolygonPoint(10, 10),
                    PolygonPoint(0, 10),
                ],
                transcript="hello worlt",
                lang="latin",
                direction=TextDirection.HORIZONTAL,
                text_type=TextType.DIALOGUE,
                detection_confidence=0.99,
                recognition_confidence=0.99,
                script_confidence=0.99,
                teacher_agreement=0.99,
            ),
            TeacherPrediction(
                asset_id="asset-ghost",
                series_id="series-a",
                image_path="img-1.jpg",
                domain=Domain.MANGA,
                polygon=[
                    PolygonPoint(50, 50),
                    PolygonPoint(60, 50),
                    PolygonPoint(60, 60),
                    PolygonPoint(50, 60),
                ],
                transcript="ghost",
                lang="latin",
                direction=TextDirection.HORIZONTAL,
                text_type=TextType.UNKNOWN,
                detection_confidence=0.99,
                recognition_confidence=0.99,
                script_confidence=0.99,
                teacher_agreement=0.99,
            ),
        ]

        summary = evaluate_predictions(gold_records, predictions, iou_threshold=0.5)

        self.assertEqual(summary["overall"]["gold_lines"], 2)
        self.assertEqual(summary["overall"]["prediction_lines"], 2)
        self.assertEqual(summary["overall"]["matched_lines"], 1)
        self.assertEqual(summary["overall"]["unmatched_gold_lines"], 1)
        self.assertEqual(summary["overall"]["unmatched_prediction_lines"], 1)
        self.assertAlmostEqual(summary["overall"]["detection_recall"], 0.5)
        self.assertAlmostEqual(summary["overall"]["detection_precision"], 0.5)
        self.assertAlmostEqual(summary["overall"]["matched_cer"], 1 / 11)
        self.assertAlmostEqual(summary["overall"]["end_to_end_cer"], 8 / 18)
        self.assertAlmostEqual(summary["slices"]["domain"]["manga"]["matched_cer"], 1 / 11)
        self.assertEqual(summary["slices"]["domain"]["webtoon"]["matched_lines"], 0)
        self.assertAlmostEqual(summary["slices"]["domain"]["webtoon"]["end_to_end_cer"], 1.0)
        self.assertEqual(summary["slices"]["split"]["test"]["gold_lines"], 1)
        self.assertEqual(summary["slices"]["direction"]["vertical"]["unmatched_gold_lines"], 1)


if __name__ == "__main__":
    unittest.main()
