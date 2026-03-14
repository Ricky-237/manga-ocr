from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from manga_ocr.domain import Domain, PageAsset, TextDirection, TextType
from manga_ocr.pseudo_label import (
    PseudoLabelThresholds,
    build_review_tasks,
    build_silver_manifest,
    read_prediction_records,
    teacher_predictions_from_records,
)
from manga_ocr.splits import SeriesSplitStrategy


class PseudoLabelWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path(".tmp-tests")
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f"workflow-{uuid4().hex}"
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _asset(self, asset_id_suffix: str, domain: Domain = Domain.MANGA) -> PageAsset:
        chapter_id = f"chapter-{asset_id_suffix}"
        page_index = int(asset_id_suffix)
        return PageAsset(
            source_id="api",
            series_id="series-1",
            chapter_id=chapter_id,
            page_index=page_index,
            image_path=f"data/raw/{asset_id_suffix}.png",
            sha256=f"sha-{asset_id_suffix}",
            phash=None,
            width=100,
            height=200,
            fetched_at="2026-03-13T00:00:00+00:00",
            domain=domain,
        )

    def test_read_prediction_records_supports_json_array(self) -> None:
        payload = [
            {
                "asset_id": "api:series-1:chapter-0:0000",
                "series_id": "series-1",
                "image_path": "data/raw/0.png",
                "domain": "manga",
                "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "transcript": "hello",
            }
        ]
        path = self.workdir / "predictions.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        records = read_prediction_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["transcript"], "hello")

    def test_teacher_predictions_can_fill_missing_fields_from_assets(self) -> None:
        asset = self._asset("0")
        records = [
            {
                "asset_id": asset.asset_id,
                "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "transcript": "bonjour",
                "lang": "latin",
                "direction": "horizontal",
                "text_type": "dialogue",
                "confidence": 0.9,
            }
        ]

        predictions = teacher_predictions_from_records(records, {asset.asset_id: asset})

        self.assertEqual(predictions[0].series_id, asset.series_id)
        self.assertEqual(predictions[0].image_path, asset.image_path)
        self.assertEqual(predictions[0].domain, asset.domain)
        self.assertAlmostEqual(predictions[0].teacher_agreement, 0.9)

    def test_build_silver_manifest_filters_predictions(self) -> None:
        asset = self._asset("0")
        records = [
            {
                "asset_id": asset.asset_id,
                "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "transcript": "good",
                "lang": "latin",
                "direction": "horizontal",
                "text_type": "dialogue",
                "detection_confidence": 0.95,
                "recognition_confidence": 0.94,
                "script_confidence": 0.93,
                "teacher_agreement": 0.92,
            },
            {
                "asset_id": asset.asset_id,
                "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "transcript": "bad",
                "lang": "latin",
                "direction": "horizontal",
                "text_type": "dialogue",
                "detection_confidence": 0.40,
                "recognition_confidence": 0.94,
                "script_confidence": 0.93,
                "teacher_agreement": 0.92,
            },
        ]

        predictions = teacher_predictions_from_records(records, {asset.asset_id: asset})
        records_out, rejected = build_silver_manifest(
            predictions,
            split_strategy=SeriesSplitStrategy(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
            thresholds=PseudoLabelThresholds(),
        )

        self.assertEqual(len(records_out), 1)
        self.assertEqual(records_out[0].split.value, "train")
        self.assertEqual(records_out[0].transcript, "good")
        self.assertEqual(len(rejected), 1)
        self.assertIn("low_detection_confidence", rejected[0].reasons)

    def test_build_review_tasks_selects_hardest_asset_and_embeds_all_predictions(self) -> None:
        easy_asset = self._asset("0")
        hard_asset = self._asset("1", domain=Domain.WEBTOON)
        records = [
            {
                "asset_id": easy_asset.asset_id,
                "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "transcript": "easy",
                "lang": "latin",
                "direction": "horizontal",
                "text_type": "dialogue",
                "detection_confidence": 0.99,
                "recognition_confidence": 0.99,
                "script_confidence": 0.99,
                "teacher_agreement": 0.99,
            },
            {
                "asset_id": hard_asset.asset_id,
                "polygon": [{"x": 10, "y": 12}, {"x": 13, "y": 14}, {"x": 15, "y": 16}],
                "transcript": "hard-1",
                "lang": "kr",
                "direction": "vertical",
                "text_type": "sfx",
                "detection_confidence": 0.70,
                "recognition_confidence": 0.78,
                "script_confidence": 0.76,
                "teacher_agreement": 0.74,
                "min_text_height": 8,
            },
            {
                "asset_id": hard_asset.asset_id,
                "polygon": [{"x": 20, "y": 22}, {"x": 23, "y": 24}, {"x": 25, "y": 26}],
                "transcript": "hard-2",
                "lang": "kr",
                "direction": "vertical",
                "text_type": "dialogue",
                "detection_confidence": 0.72,
                "recognition_confidence": 0.76,
                "script_confidence": 0.74,
                "teacher_agreement": 0.70,
                "min_text_height": 9,
            },
        ]

        predictions = teacher_predictions_from_records(
            records,
            {easy_asset.asset_id: easy_asset, hard_asset.asset_id: hard_asset},
        )
        tasks, decisions = build_review_tasks(
            [easy_asset, hard_asset],
            predictions,
            page_limit=1,
            thresholds=PseudoLabelThresholds(),
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["data"]["asset_id"], hard_asset.asset_id)
        self.assertEqual(len(tasks[0]["predictions"][0]["result"]), 8)
        self.assertEqual({decision.prediction.asset_id for decision in decisions}, {hard_asset.asset_id})
        self.assertEqual(decisions[0].prediction.direction, TextDirection.VERTICAL)
        self.assertIn(decisions[0].prediction.text_type, {TextType.SFX, TextType.DIALOGUE})


if __name__ == "__main__":
    unittest.main()
