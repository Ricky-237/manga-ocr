from __future__ import annotations

import base64
import io
import json
import os
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from manga_ocr.domain import Domain, PageAsset
from manga_ocr.ingest import ChapterApiAdapter, ChapterDescriptor, ChapterListing, PageRef
from manga_ocr.manifests import read_jsonl
from manga_ocr.storage import CatalogDatabase

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZkvoAAAAASUVORK5CYII="
)


class FakeConsumetCliAdapter(ChapterApiAdapter):
    def __init__(self) -> None:
        self.latest = [
            ChapterDescriptor(
                chapter_id="manga_a/v1/c10",
                series_id="manga_a",
                source_id="consumet-mangahere",
                metadata={"title": "Manga A", "last_chapter": "Ch.10"},
            ),
            ChapterDescriptor(
                chapter_id="manga_b/v1/c20",
                series_id="manga_b",
                source_id="consumet-mangahere",
                metadata={"title": "Manga B", "last_chapter": "Ch.20"},
            ),
        ]
        self.manga_chapters = [
            ChapterDescriptor(
                chapter_id="one_piece/v98/c1176",
                series_id="one_piece",
                source_id="consumet-mangahere",
                metadata={"manga_title": "One Piece", "chapter_title": "Vol.98 Ch.1176"},
            ),
            ChapterDescriptor(
                chapter_id="one_piece/v98/c1175",
                series_id="one_piece",
                source_id="consumet-mangahere",
                metadata={"manga_title": "One Piece", "chapter_title": "Vol.98 Ch.1175"},
            ),
        ]

    def listChapters(self, cursor: str | None = None) -> ChapterListing:
        return ChapterListing(chapters=list(self.latest), next_cursor="2")

    def list_latest_chapters(self, page: int = 1, limit: int | None = None) -> ChapterListing:
        chapters = self.latest if limit is None else self.latest[:limit]
        return ChapterListing(chapters=list(chapters), next_cursor=str(page + 1))

    def get_manga_chapters(self, manga_id: str, limit: int | None = None):
        chapters = self.manga_chapters if limit is None else self.manga_chapters[:limit]
        return list(chapters)

    def getChapterPages(self, chapter_id: str):
        return [
            PageRef(
                source_id="consumet-mangahere",
                series_id=chapter_id.split("/", 1)[0],
                chapter_id=chapter_id,
                page_index=0,
                image_ref=f"https://img/{chapter_id}.jpg",
                domain=Domain.MANGA,
            )
        ]

    def downloadPage(self, page_ref: PageRef) -> bytes:
        return PNG_1X1


class FakeFlakyConsumetCliAdapter(FakeConsumetCliAdapter):
    def downloadPage(self, page_ref: PageRef) -> bytes:
        if page_ref.chapter_id == "manga_b/v1/c20":
            raise TimeoutError("The read operation timed out")
        return PNG_1X1


class FakeTeacherPredictor:
    def predict_page(self, asset: PageAsset):
        return [
            {
                "polygon": [
                    {"x": 0, "y": 0},
                    {"x": 32, "y": 0},
                    {"x": 32, "y": 16},
                    {"x": 0, "y": 16},
                ],
                "transcript": f"ocr-{asset.series_id}",
                "lang": "ja",
                "direction": "vertical",
                "text_type": "dialogue",
                "detection_confidence": 0.97,
                "recognition_confidence": 0.96,
                "script_confidence": 0.95,
                "teacher_agreement": 0.94,
            }
        ]


class FakeMixedTeacherPredictor:
    def predict_page(self, asset: PageAsset):
        base_prediction = {
            "polygon": [
                {"x": 0, "y": 0},
                {"x": 40, "y": 0},
                {"x": 40, "y": 20},
                {"x": 0, "y": 20},
            ],
            "transcript": f"ocr-{asset.series_id}",
            "lang": "mixed",
            "direction": "horizontal",
            "text_type": "dialogue",
            "detection_confidence": 0.97,
            "recognition_confidence": 0.96,
            "script_confidence": 0.95,
            "teacher_agreement": 0.94,
            "min_text_height": 20,
        }
        if asset.series_id == "series-b":
            base_prediction.update(
                {
                    "transcript": "tiny",
                    "detection_confidence": 0.40,
                    "recognition_confidence": 0.45,
                    "script_confidence": 0.42,
                    "teacher_agreement": 0.30,
                    "min_text_height": 6,
                }
            )
        return [base_prediction]


class CliCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path(".tmp-tests")
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f"cli-{uuid4().hex}"
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _run_cli(self, argv: list[str], adapter: ChapterApiAdapter | None = None) -> dict:
        from manga_ocr import cli

        fake_adapter = adapter or FakeConsumetCliAdapter()
        stdout = io.StringIO()
        with patch("sys.argv", argv), patch("manga_ocr.cli.ConsumetMangahereAdapter", return_value=fake_adapter):
            with redirect_stdout(stdout):
                cli.main()
        return json.loads(stdout.getvalue())

    def _insert_asset(self, db_path: Path, series_id: str, chapter_id: str, page_index: int) -> PageAsset:
        catalog = CatalogDatabase(db_path)
        catalog.initialize()
        asset = PageAsset(
            source_id="consumet-mangahere",
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            image_path=str(self.workdir / f"{series_id}-{page_index}.jpg"),
            sha256=f"sha-{series_id}-{page_index}",
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )
        catalog.upsert_page_asset(asset)
        return asset

    def test_ingest_latest_command_respects_limit(self) -> None:
        payload = self._run_cli(
            [
                "manga-ocr",
                "ingest-latest",
                "--db",
                str(self.workdir / "catalog.sqlite"),
                "--storage-root",
                str(self.workdir / "data"),
                "--page",
                "3",
                "--limit",
                "1",
            ]
        )

        self.assertEqual(payload["page"], 3)
        self.assertEqual(payload["requested_chapters"], 1)
        self.assertEqual(len(payload["chapters"]), 1)
        self.assertEqual(payload["chapters"][0]["series_id"], "manga_a")
        self.assertEqual(payload["chapters"][0]["pages"], 1)
        self.assertEqual(payload["succeeded_chapters"], 1)
        self.assertEqual(payload["failed_chapters"], 0)

    def test_ingest_manga_command_defaults_to_single_latest_chapter(self) -> None:
        payload = self._run_cli(
            [
                "manga-ocr",
                "ingest-manga",
                "--db",
                str(self.workdir / "catalog.sqlite"),
                "--storage-root",
                str(self.workdir / "data"),
                "--manga-id",
                "one_piece",
            ]
        )

        self.assertEqual(payload["manga_id"], "one_piece")
        self.assertFalse(payload["all_chapters"])
        self.assertEqual(payload["requested_chapters"], 1)
        self.assertEqual(payload["chapters"][0]["chapter_id"], "one_piece/v98/c1176")

    def test_ingest_manga_command_can_ingest_all_chapters(self) -> None:
        payload = self._run_cli(
            [
                "manga-ocr",
                "ingest-manga",
                "--db",
                str(self.workdir / "catalog.sqlite"),
                "--storage-root",
                str(self.workdir / "data"),
                "--manga-id",
                "one_piece",
                "--all-chapters",
            ]
        )

        self.assertTrue(payload["all_chapters"])
        self.assertEqual(payload["requested_chapters"], 2)
        self.assertEqual(len(payload["chapters"]), 2)

    def test_ingest_latest_continues_and_reports_failed_chapters(self) -> None:
        payload = self._run_cli(
            [
                "manga-ocr",
                "ingest-latest",
                "--db",
                str(self.workdir / "catalog.sqlite"),
                "--storage-root",
                str(self.workdir / "data"),
                "--page",
                "1",
                "--limit",
                "2",
            ],
            adapter=FakeFlakyConsumetCliAdapter(),
        )

        self.assertEqual(payload["requested_chapters"], 2)
        self.assertEqual(payload["succeeded_chapters"], 1)
        self.assertEqual(payload["failed_chapters"], 1)
        self.assertEqual(payload["chapters"][0]["status"], "ok")
        self.assertEqual(payload["chapters"][1]["status"], "failed")
        self.assertEqual(payload["chapters"][1]["pages"], 0)
        self.assertEqual(payload["chapters"][1]["page_errors"][0]["stage"], "download_page")
        self.assertIn("timed out", payload["chapters"][1]["page_errors"][0]["error"].lower())

    def test_run_teacher_command_exports_predictions(self) -> None:
        from manga_ocr import cli

        db_path = self.workdir / "catalog.sqlite"
        output_path = self.workdir / "teacher_predictions.jsonl"
        asset = self._insert_asset(db_path, series_id="series-a", chapter_id="series-a/c1", page_index=0)
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "manga-ocr",
                "run-teacher",
                "--db",
                str(db_path),
                "--out",
                str(output_path),
                "--predictor",
                "tests.fake:Predictor",
                "--series-id",
                "series-a",
            ],
        ), patch("manga_ocr.cli.load_teacher_predictor", return_value=FakeTeacherPredictor()):
            with redirect_stdout(stdout):
                cli.main()

        payload = json.loads(stdout.getvalue())
        records = read_jsonl(output_path)
        self.assertEqual(payload["selected_assets"], 1)
        self.assertEqual(payload["predictions"], 1)
        self.assertEqual(payload["failed_assets"], 0)
        self.assertEqual(records[0]["asset_id"], asset.asset_id)
        self.assertEqual(records[0]["transcript"], "ocr-series-a")
        self.assertEqual(records[0]["direction"], "vertical")

    def test_run_teacher_cycle_exports_predictions_silver_and_review(self) -> None:
        from manga_ocr import cli

        db_path = self.workdir / "catalog.sqlite"
        predictions_path = self.workdir / "teacher_predictions.jsonl"
        silver_path = self.workdir / "silver_manifest.jsonl"
        rejected_path = self.workdir / "rejected.jsonl"
        review_path = self.workdir / "review_tasks.json"
        review_decisions_path = self.workdir / "review_decisions.jsonl"
        asset_a = self._insert_asset(db_path, series_id="series-a", chapter_id="series-a/c1", page_index=0)
        asset_b = self._insert_asset(db_path, series_id="series-b", chapter_id="series-b/c1", page_index=0)
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "manga-ocr",
                "run-teacher-cycle",
                "--db",
                str(db_path),
                "--predictions-out",
                str(predictions_path),
                "--silver-out",
                str(silver_path),
                "--review-out",
                str(review_path),
                "--rejected-out",
                str(rejected_path),
                "--review-decisions-out",
                str(review_decisions_path),
                "--predictor",
                "tests.fake:Predictor",
                "--limit",
                "10",
                "--continue-on-error",
            ],
        ), patch("manga_ocr.cli.load_teacher_predictor", return_value=FakeMixedTeacherPredictor()):
            with redirect_stdout(stdout):
                cli.main()

        payload = json.loads(stdout.getvalue())
        prediction_records = read_jsonl(predictions_path)
        silver_records = read_jsonl(silver_path)
        rejected_records = read_jsonl(rejected_path)
        review_decisions = read_jsonl(review_decisions_path)
        review_tasks = json.loads(review_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["selected_assets"], 2)
        self.assertEqual(payload["input_predictions"], 2)
        self.assertEqual(payload["accepted_silver"], 1)
        self.assertEqual(payload["review_candidates"], 1)
        self.assertEqual(payload["review_tasks"], 2)
        self.assertEqual(len(prediction_records), 2)
        self.assertEqual(len(silver_records), 1)
        self.assertEqual(silver_records[0]["series_id"], asset_a.series_id)
        self.assertEqual(len(rejected_records), 1)
        self.assertEqual(rejected_records[0]["prediction"]["asset_id"], asset_b.asset_id)
        self.assertEqual(len(review_decisions), 2)
        self.assertEqual(len(review_tasks), 2)

    def test_convert_label_studio_export_resolves_repo_relative_input_from_src_cwd(self) -> None:
        from manga_ocr import cli

        repo_root = (self.workdir / "repo").resolve()
        (repo_root / "annotation" / "exports").mkdir(parents=True)
        (repo_root / "src").mkdir(parents=True)
        export_path = repo_root / "annotation" / "exports" / "reviewed.json"
        output_path = repo_root / "src" / "manifests" / "dataset_manifest.jsonl"
        export_path.write_text(
            json.dumps(
                [
                    {
                        "data": {
                            "image_path": "img-1.jpg",
                            "series_id": "series-a",
                            "domain": "manga",
                            "width": 100,
                            "height": 200,
                        },
                        "annotations": [
                            {
                                "result": [
                                    {
                                        "id": "region-1",
                                        "from_name": "text_region",
                                        "to_name": "image",
                                        "type": "polygonlabels",
                                        "original_width": 100,
                                        "original_height": 200,
                                        "value": {
                                            "points": [[10, 10], [20, 10], [20, 20]],
                                            "polygonlabels": ["dialogue"],
                                        },
                                    },
                                    {
                                        "id": "region-1-text",
                                        "parentID": "region-1",
                                        "from_name": "transcript",
                                        "to_name": "image",
                                        "type": "textarea",
                                        "value": {"text": ["Bonjour"]},
                                    },
                                    {
                                        "id": "region-1-lang",
                                        "parentID": "region-1",
                                        "from_name": "language",
                                        "to_name": "image",
                                        "type": "choices",
                                        "value": {"choices": ["latin"]},
                                    },
                                    {
                                        "id": "region-1-direction",
                                        "parentID": "region-1",
                                        "from_name": "direction",
                                        "to_name": "image",
                                        "type": "choices",
                                        "value": {"choices": ["horizontal"]},
                                    },
                                ]
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        stdout = io.StringIO()
        original_cwd = Path.cwd()
        try:
            os.chdir(repo_root / "src")
            with patch("sys.argv", [
                "manga-ocr",
                "convert-label-studio-export",
                "--input",
                "annotation/exports/reviewed.json",
                "--out",
                "manifests/dataset_manifest.jsonl",
            ]), patch("manga_ocr.cli._REPO_ROOT", repo_root):
                with redirect_stdout(stdout):
                    cli.main()
        finally:
            os.chdir(original_cwd)

        self.assertTrue(output_path.exists())
        records = read_jsonl(output_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["transcript"], "Bonjour")

    def test_evaluate_predictions_command_outputs_report(self) -> None:
        from manga_ocr import cli

        gold_path = self.workdir / "dataset_manifest.jsonl"
        predictions_path = self.workdir / "teacher_predictions.jsonl"
        report_path = self.workdir / "eval_report.json"
        gold_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "image_path": "img-1.jpg",
                            "series_id": "series-a",
                            "domain": "manga",
                            "tile_id": None,
                            "polygon": [
                                {"x": 0, "y": 0},
                                {"x": 10, "y": 0},
                                {"x": 10, "y": 10},
                                {"x": 0, "y": 10},
                            ],
                            "transcript": "hello world",
                            "lang": "latin",
                            "direction": "horizontal",
                            "text_type": "dialogue",
                            "split": "train",
                        }
                    ),
                    json.dumps(
                        {
                            "image_path": "img-2.jpg",
                            "series_id": "series-b",
                            "domain": "webtoon",
                            "tile_id": None,
                            "polygon": [
                                {"x": 0, "y": 0},
                                {"x": 10, "y": 0},
                                {"x": 10, "y": 10},
                                {"x": 0, "y": 10},
                            ],
                            "transcript": "bonjour",
                            "lang": "latin",
                            "direction": "vertical",
                            "text_type": "caption",
                            "split": "test",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "asset_id": "asset-1",
                            "series_id": "series-a",
                            "image_path": "img-1.jpg",
                            "domain": "manga",
                            "polygon": [
                                {"x": 0, "y": 0},
                                {"x": 10, "y": 0},
                                {"x": 10, "y": 10},
                                {"x": 0, "y": 10},
                            ],
                            "transcript": "hello worlt",
                            "lang": "latin",
                            "direction": "horizontal",
                            "text_type": "dialogue",
                            "detection_confidence": 0.97,
                            "recognition_confidence": 0.96,
                            "script_confidence": 0.95,
                            "teacher_agreement": 0.94,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "manga-ocr",
                "evaluate-predictions",
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
                "--iou-threshold",
                "0.5",
                "--out",
                str(report_path),
            ],
        ):
            with redirect_stdout(stdout):
                cli.main()

        payload = json.loads(stdout.getvalue())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["overall"]["gold_lines"], 2)
        self.assertEqual(payload["overall"]["matched_lines"], 1)
        self.assertAlmostEqual(payload["overall"]["detection_recall"], 0.5)
        self.assertAlmostEqual(payload["overall"]["end_to_end_cer"], 8 / 18)
        self.assertEqual(report["slices"]["domain"]["webtoon"]["matched_lines"], 0)
        self.assertEqual(payload["output"], str(report_path))


if __name__ == "__main__":
    unittest.main()



