from __future__ import annotations

import shutil
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from manga_ocr.domain import Domain, PageAsset
from manga_ocr.teachers import TesseractTeacherPredictor


TSV_SAMPLE = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t96\tHello
5\t1\t1\t1\t1\t2\t45\t20\t35\t12\t93\tworld
5\t1\t1\t1\t2\t1\t12\t40\t20\t14\t88\tこんにちは
"""


class TesseractTeacherPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path(".tmp-tests")
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f"tesseract-{uuid4().hex}"
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _asset(self) -> PageAsset:
        return PageAsset(
            source_id="consumet-mangahere",
            series_id="series-a",
            chapter_id="series-a/c1",
            page_index=0,
            image_path=str(self.workdir / "page.jpg"),
            sha256="sha-page",
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )

    def test_predict_page_parses_tsv_into_line_predictions(self) -> None:
        predictor = TesseractTeacherPredictor()
        completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout=TSV_SAMPLE,
            stderr="",
        )

        with patch("manga_ocr.teachers.tesseract.subprocess.run", return_value=completed) as run_mock:
            predictions = predictor.predict_page(self._asset())

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["transcript"], "Hello world")
        self.assertEqual(predictions[0]["lang"], "latin")
        self.assertEqual(predictions[0]["direction"], "horizontal")
        self.assertAlmostEqual(predictions[0]["recognition_confidence"], 0.945)
        self.assertEqual(predictions[1]["transcript"], "こんにちは")
        self.assertEqual(predictions[1]["lang"], "ja")
        self.assertEqual(predictions[1]["text_type"], "dialogue")
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 120.0)

    def test_predict_page_raises_clean_error_when_tesseract_missing(self) -> None:
        predictor = TesseractTeacherPredictor(executable="missing-tesseract")

        with patch("manga_ocr.teachers.tesseract.subprocess.run", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "Tesseract executable not found"):
                predictor.predict_page(self._asset())

    def test_predict_page_skips_low_confidence_lines(self) -> None:
        predictor = TesseractTeacherPredictor(min_line_confidence=0.90, min_word_confidence=10.0)
        completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout="""level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t40\tweak
""",
            stderr="",
        )

        with patch("manga_ocr.teachers.tesseract.subprocess.run", return_value=completed):
            predictions = predictor.predict_page(self._asset())

        self.assertEqual(predictions, [])


if __name__ == "__main__":
    unittest.main()
