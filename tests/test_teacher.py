from __future__ import annotations

import json
import os
import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from manga_ocr.domain import Domain, PageAsset
from manga_ocr.manifests import read_jsonl
from manga_ocr.pseudo_label import (
    TeacherPredictor,
    load_teacher_predictor,
    run_teacher_predictions,
    select_page_assets,
    write_teacher_predictions,
)
from manga_ocr.teachers import TesseractTeacherPredictor, YoloOnnxTesseractTeacherPredictor


class InlinePredictor(TeacherPredictor):
    def predict_page(self, asset: PageAsset):
        return [
            {
                "polygon": [
                    {"x": 0, "y": 0},
                    {"x": 12, "y": 0},
                    {"x": 12, "y": 10},
                    {"x": 0, "y": 10},
                ],
                "transcript": f"text-{asset.page_index}",
                "detection_confidence": 0.95,
                "recognition_confidence": 0.94,
                "script_confidence": 0.93,
                "teacher_agreement": 0.92,
            }
        ]


class FlakyPredictor(TeacherPredictor):
    def predict_page(self, asset: PageAsset):
        if asset.page_index == 1:
            raise RuntimeError("teacher backend crashed")
        return InlinePredictor().predict_page(asset)


class TeacherInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path(".tmp-tests")
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f"teacher-{uuid4().hex}"
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _asset(self, series_id: str, chapter_id: str, page_index: int, is_duplicate: bool = False) -> PageAsset:
        return PageAsset(
            source_id="consumet-mangahere",
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            image_path=str(self.workdir / f"{series_id}-{chapter_id}-{page_index}.jpg"),
            sha256=f"sha-{series_id}-{chapter_id}-{page_index}",
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
            is_duplicate=is_duplicate,
        )

    def test_load_teacher_predictor_imports_builtin_tesseract_class(self) -> None:
        config_path = self.workdir / "teacher-config.json"
        config_path.write_text(json.dumps({"languages": "eng", "preserve_spaces": False}), encoding="utf-8")

        predictor = load_teacher_predictor(
            "manga_ocr.teachers.tesseract:TesseractTeacherPredictor",
            str(config_path),
        )

        self.assertIsInstance(predictor, TesseractTeacherPredictor)
        self.assertEqual(predictor.languages, "eng")
        self.assertFalse(predictor.preserve_spaces)

    def test_load_teacher_predictor_resolves_repo_relative_config_from_src_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        try:
            os.chdir(repo_root / "src")
            predictor = load_teacher_predictor(
                "manga_ocr.teachers.tesseract:TesseractTeacherPredictor",
                "configs/teacher_tesseract.json",
            )
        finally:
            os.chdir(original_cwd)

        self.assertIsInstance(predictor, TesseractTeacherPredictor)
        self.assertEqual(predictor.languages, "jpn+kor+eng")

    def test_load_teacher_predictor_imports_yolo_tesseract_backend(self) -> None:
        config_path = self.workdir / "teacher-yolo-config.json"
        config_path.write_text(
            json.dumps({
                "detector_model_path": "D:/engine-fault-detection/manga109_yolo/yolo26n.onnx",
                "crop_padding": 12,
                "recognizer": {"languages": "eng"},
            }),
            encoding="utf-8",
        )

        predictor = load_teacher_predictor(
            "manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor",
            str(config_path),
        )

        self.assertIsInstance(predictor, YoloOnnxTesseractTeacherPredictor)
        self.assertEqual(predictor.crop_padding, 12)
        self.assertEqual(predictor.recognizer.languages, "eng")

    def test_select_page_assets_filters_and_skips_duplicates(self) -> None:
        assets = [
            self._asset("series-a", "series-a/c1", 0),
            self._asset("series-a", "series-a/c1", 1, is_duplicate=True),
            self._asset("series-b", "series-b/c2", 0),
        ]

        selected = select_page_assets(assets, series_id="series-a")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].series_id, "series-a")
        self.assertFalse(selected[0].is_duplicate)

    def test_run_teacher_predictions_normalizes_dicts_and_writes_jsonl(self) -> None:
        assets = [self._asset("series-a", "series-a/c1", 0)]

        predictions, failures = run_teacher_predictions(InlinePredictor(), assets)
        output_path = self.workdir / "teacher_predictions.jsonl"
        write_teacher_predictions(predictions, output_path)
        records = read_jsonl(output_path)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(failures), 0)
        self.assertEqual(records[0]["asset_id"], assets[0].asset_id)
        self.assertEqual(records[0]["series_id"], "series-a")
        self.assertEqual(records[0]["image_path"], assets[0].image_path)
        self.assertEqual(records[0]["transcript"], "text-0")
        self.assertEqual(records[0]["direction"], "horizontal")
        self.assertEqual(records[0]["text_type"], "unknown")

    def test_run_teacher_predictions_collects_failures_when_continuing(self) -> None:
        assets = [
            self._asset("series-a", "series-a/c1", 0),
            self._asset("series-a", "series-a/c1", 1),
        ]

        predictions, failures = run_teacher_predictions(FlakyPredictor(), assets, continue_on_error=True)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].asset_id, assets[1].asset_id)
        self.assertIn("crashed", failures[0].error)


if __name__ == "__main__":
    unittest.main()
