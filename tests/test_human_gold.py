from __future__ import annotations

import unittest
from datetime import datetime, timezone

from manga_ocr.annotation import build_human_gold_tasks, select_human_gold_pages, summarize_human_gold_batch
from manga_ocr.domain import DatasetSplit, Domain, PageAsset, PolygonPoint, TextDirection, TextType
from manga_ocr.pseudo_label import PseudoLabelThresholds, TeacherPrediction


class FixedSplitStrategy:
    train_ratio = 0.34
    val_ratio = 0.33
    test_ratio = 0.33

    def __init__(self, mapping: dict[str, DatasetSplit]) -> None:
        self._mapping = mapping

    def assign(self, series_id: str) -> DatasetSplit:
        return self._mapping[series_id]


class HumanGoldTests(unittest.TestCase):
    def _make_asset(self, series_id: str, chapter_id: str, page_index: int) -> PageAsset:
        return PageAsset(
            source_id="consumet-mangahere",
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            image_path=f"data/{series_id}-{page_index}.jpg",
            sha256=f"sha-{series_id}-{page_index}",
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )

    def _make_prediction(
        self,
        asset: PageAsset,
        transcript: str,
        *,
        detection_confidence: float = 0.97,
        recognition_confidence: float = 0.96,
        script_confidence: float = 0.95,
        teacher_agreement: float = 0.94,
        min_text_height: float | None = 18.0,
    ) -> TeacherPrediction:
        return TeacherPrediction(
            asset_id=asset.asset_id,
            series_id=asset.series_id,
            image_path=asset.image_path,
            domain=asset.domain,
            polygon=[
                PolygonPoint(10, 10),
                PolygonPoint(60, 10),
                PolygonPoint(60, 40),
                PolygonPoint(10, 40),
            ],
            transcript=transcript,
            lang="ja",
            direction=TextDirection.VERTICAL,
            text_type=TextType.DIALOGUE,
            detection_confidence=detection_confidence,
            recognition_confidence=recognition_confidence,
            script_confidence=script_confidence,
            teacher_agreement=teacher_agreement,
            min_text_height=min_text_height,
        )

    def test_select_human_gold_pages_applies_split_and_review_policy(self) -> None:
        train_asset = self._make_asset("series-train", "series-train/c1", 0)
        val_asset = self._make_asset("series-val", "series-val/c1", 0)
        test_asset = self._make_asset("series-test", "series-test/c1", 0)
        split_strategy = FixedSplitStrategy(
            {
                "series-train": DatasetSplit.TRAIN,
                "series-val": DatasetSplit.VAL,
                "series-test": DatasetSplit.TEST,
            }
        )
        predictions = [
            self._make_prediction(train_asset, "easy line"),
            self._make_prediction(
                val_asset,
                "hard line",
                detection_confidence=0.35,
                recognition_confidence=0.45,
                script_confidence=0.40,
                teacher_agreement=0.25,
                min_text_height=6.0,
            ),
        ]

        selections = select_human_gold_pages(
            [train_asset, val_asset, test_asset],
            predictions=predictions,
            page_limit=3,
            split_strategy=split_strategy,
            thresholds=PseudoLabelThresholds(),
        )

        self.assertEqual(len(selections), 3)
        self.assertEqual(selections[0].asset.asset_id, val_asset.asset_id)
        selected_by_series = {selection.asset.series_id: selection for selection in selections}
        self.assertEqual(selected_by_series["series-train"].review_mode, "single_review")
        self.assertEqual(selected_by_series["series-val"].review_mode, "single_review")
        self.assertEqual(selected_by_series["series-test"].review_mode, "double_review")
        self.assertEqual(selected_by_series["series-test"].preannotated_lines, 0)

        summary = summarize_human_gold_batch(selections)
        self.assertEqual(summary["selected_pages"], 3)
        self.assertEqual(summary["preannotated_pages"], 2)
        self.assertEqual(summary["split_counts"], {"test": 1, "train": 1, "val": 1})
        self.assertEqual(summary["review_mode_counts"], {"double_review": 1, "single_review": 2})

    def test_build_human_gold_tasks_embeds_metadata_and_preannotations(self) -> None:
        train_asset = self._make_asset("series-train", "series-train/c1", 0)
        test_asset = self._make_asset("series-test", "series-test/c1", 0)
        split_strategy = FixedSplitStrategy(
            {
                "series-train": DatasetSplit.TRAIN,
                "series-test": DatasetSplit.TEST,
            }
        )
        predictions = [
            self._make_prediction(train_asset, "bonjour"),
            self._make_prediction(
                test_asset,
                "wow",
                detection_confidence=0.50,
                recognition_confidence=0.55,
                script_confidence=0.52,
                teacher_agreement=0.40,
                min_text_height=8.0,
            ),
        ]
        selections = select_human_gold_pages(
            [train_asset, test_asset],
            predictions=predictions,
            page_limit=2,
            split_strategy=split_strategy,
        )

        tasks = build_human_gold_tasks(
            selections,
            predictions=predictions,
            image_base_url="https://labels.example/assets",
        )

        self.assertEqual(len(tasks), 2)
        task_by_asset_id = {task["data"]["asset_id"]: task for task in tasks}
        train_task = task_by_asset_id[train_asset.asset_id]
        test_task = task_by_asset_id[test_asset.asset_id]

        self.assertEqual(train_task["data"]["target_split"], "train")
        self.assertEqual(train_task["data"]["review_mode"], "single_review")
        self.assertEqual(train_task["data"]["gold_batch_rank"], 2)
        self.assertEqual(train_task["data"]["preannotated_lines"], 1)
        self.assertTrue(train_task["data"]["image"].startswith("https://labels.example/assets/"))
        self.assertEqual(len(train_task["predictions"][0]["result"]), 4)

        self.assertEqual(test_task["data"]["target_split"], "test")
        self.assertEqual(test_task["data"]["review_mode"], "double_review")
        self.assertEqual(test_task["data"]["gold_batch_rank"], 1)
        self.assertIn("small_text", test_task["data"]["priority_reasons"])
        self.assertEqual(len(test_task["predictions"][0]["result"]), 4)


if __name__ == "__main__":
    unittest.main()
