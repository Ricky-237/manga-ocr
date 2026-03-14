from __future__ import annotations

import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from manga_ocr.detection.yolo_onnx import TextDetection
from manga_ocr.domain import Domain, PageAsset, PolygonPoint, TextDirection
from manga_ocr.teachers import YoloOnnxTesseractTeacherPredictor


class _FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)

    def detect_path(self, image_path: str):
        return list(self._detections)


class _FakeRecognizer:
    def __init__(self, predictions):
        self._predictions = list(predictions)

    def predict_page(self, asset: PageAsset):
        return list(self._predictions)


class _FakeCropImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def save(self, output_path: Path) -> None:
        Path(output_path).write_bytes(b'fake-crop')


class _FakePageImage:
    def crop(self, crop_bounds):
        left, top, right, bottom = crop_bounds
        return _FakeCropImage(right - left, bottom - top)

    def close(self) -> None:
        return None


class _StubYoloTeacher(YoloOnnxTesseractTeacherPredictor):
    def _open_image(self, image_path: str | Path):
        return _FakePageImage()

    @staticmethod
    def _save_crop(image, output_path: Path) -> None:
        return None


class YoloOnnxTesseractTeacherPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path('.tmp-tests')
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f'yolo-teacher-{uuid4().hex}'
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _asset(self, width: int = 800, height: int = 1200) -> PageAsset:
        return PageAsset(
            source_id='consumet-mangahere',
            series_id='series-a',
            chapter_id='series-a/c1',
            page_index=0,
            image_path=str(self.workdir / 'page.jpg'),
            sha256='sha-page',
            phash=None,
            width=width,
            height=height,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )

    def test_predict_page_projects_crop_predictions_back_to_page_space(self) -> None:
        detector = _FakeDetector(
            [
                TextDetection(
                    polygon=[
                        PolygonPoint(100.0, 50.0),
                        PolygonPoint(160.0, 50.0),
                        PolygonPoint(160.0, 120.0),
                        PolygonPoint(100.0, 120.0),
                    ],
                    confidence=0.97,
                )
            ]
        )
        recognizer = _FakeRecognizer(
            [
                {
                    'polygon': [
                        {'x': 0.0, 'y': 0.0},
                        {'x': 40.0, 'y': 0.0},
                        {'x': 40.0, 'y': 20.0},
                        {'x': 0.0, 'y': 20.0},
                    ],
                    'transcript': 'Hello world',
                    'lang': 'latin',
                    'recognition_confidence': 0.88,
                    'script_confidence': 0.93,
                    'text_type': 'dialogue',
                }
            ]
        )
        predictor = _StubYoloTeacher(
            detector_model_path='dummy.onnx',
            detector=detector,
            recognizer_backend=recognizer,
            crop_padding=10,
        )

        predictions = predictor.predict_page(self._asset())

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertEqual(prediction['asset_id'], 'consumet-mangahere:series-a:series-a/c1:0000')
        self.assertEqual(prediction['transcript'], 'Hello world')
        self.assertEqual(prediction['direction'], 'horizontal')
        self.assertEqual(prediction['detection_confidence'], 0.97)
        self.assertEqual(prediction['teacher_agreement'], 0.88)
        self.assertEqual(
            prediction['polygon'],
            [
                {'x': 90.0, 'y': 40.0},
                {'x': 130.0, 'y': 40.0},
                {'x': 130.0, 'y': 60.0},
                {'x': 90.0, 'y': 60.0},
            ],
        )

    def test_infer_direction_marks_tall_regions_as_vertical(self) -> None:
        predictor = _StubYoloTeacher(
            detector_model_path='dummy.onnx',
            detector=_FakeDetector([]),
            recognizer_backend=_FakeRecognizer([]),
            aspect_ratio_vertical_threshold=1.5,
        )
        detection = TextDetection(
            polygon=[
                PolygonPoint(10.0, 10.0),
                PolygonPoint(40.0, 10.0),
                PolygonPoint(40.0, 100.0),
                PolygonPoint(10.0, 100.0),
            ],
            confidence=0.9,
        )

        direction = predictor._infer_direction(detection)

        self.assertEqual(direction, TextDirection.VERTICAL)


if __name__ == '__main__':
    unittest.main()
