from __future__ import annotations

import unittest

from manga_ocr.domain import Domain, PolygonPoint, TextDirection, TextType
from manga_ocr.pseudo_label import (
    ExceptionReviewPolicy,
    PseudoLabelThresholds,
    TeacherPrediction,
    filter_for_silver,
    review_queue,
)


class PseudoLabelTests(unittest.TestCase):
    def _prediction(
        self,
        *,
        asset_id: str,
        detection: float,
        recognition: float,
        script: float,
        agreement: float,
        domain: Domain = Domain.MANGA,
        direction: TextDirection = TextDirection.HORIZONTAL,
        text_type: TextType = TextType.DIALOGUE,
        min_height: float = 16.0,
    ) -> TeacherPrediction:
        return TeacherPrediction(
            asset_id=asset_id,
            series_id="series-1",
            image_path="data/raw/sample.png",
            domain=domain,
            polygon=[PolygonPoint(0, 0), PolygonPoint(10, 0), PolygonPoint(10, 10)],
            transcript="test",
            lang="jp",
            direction=direction,
            text_type=text_type,
            detection_confidence=detection,
            recognition_confidence=recognition,
            script_confidence=script,
            teacher_agreement=agreement,
            min_text_height=min_height,
        )

    def test_filter_for_silver_keeps_only_confident_predictions(self) -> None:
        thresholds = PseudoLabelThresholds()
        confident = self._prediction(asset_id="good", detection=0.95, recognition=0.93, script=0.92, agreement=0.90)
        weak = self._prediction(asset_id="bad", detection=0.60, recognition=0.90, script=0.90, agreement=0.90)

        accepted, rejected = filter_for_silver([confident, weak], thresholds=thresholds)

        self.assertEqual([item.asset_id for item in accepted], ["good"])
        self.assertEqual(len(rejected), 1)
        self.assertIn("low_detection_confidence", rejected[0].reasons)

    def test_review_queue_prioritizes_hard_cases(self) -> None:
        easy = self._prediction(asset_id="easy", detection=0.95, recognition=0.95, script=0.95, agreement=0.95)
        hard = self._prediction(
            asset_id="hard",
            detection=0.80,
            recognition=0.78,
            script=0.70,
            agreement=0.72,
            domain=Domain.WEBTOON,
            direction=TextDirection.VERTICAL,
            text_type=TextType.SFX,
            min_height=8,
        )

        queue = review_queue([easy, hard], limit=2)

        self.assertEqual(queue[0].prediction.asset_id, "hard")
        self.assertGreater(queue[0].review_score, queue[1].review_score)

    def test_exception_review_policy_rejects_invalid_audit_rate(self) -> None:
        with self.assertRaises(ValueError):
            ExceptionReviewPolicy(audit_rate=1.5)


if __name__ == "__main__":
    unittest.main()
