from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..detection import TextDetection, YoloOnnxTextDetector
from ..domain import PageAsset, PolygonPoint, TextDirection, TextType
from ..pseudo_label.teacher import TeacherPredictor
from .tesseract import TesseractTeacherPredictor

try:  # Optional runtime dependency.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised through clean runtime errors
    Image = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]


class YoloOnnxTesseractTeacherPredictor(TeacherPredictor):
    def __init__(
        self,
        detector_model_path: str | Path,
        detector_input_size: int = 1280,
        detector_confidence_threshold: float = 0.15,
        detector_iou_threshold: float = 0.30,
        detector_min_box_size: float = 4.0,
        detector_box_format: str = 'xyxy',
        detector_padding_value: int = 127,
        detector_providers: list[str] | None = None,
        crop_padding: int = 8,
        min_crop_size: int = 8,
        aspect_ratio_vertical_threshold: float = 1.35,
        rotate_vertical_crops: bool = False,
        default_text_type: str = 'dialogue',
        crop_temp_root: str | Path | None = None,
        recognizer: dict[str, Any] | None = None,
        detector: YoloOnnxTextDetector | None = None,
        recognizer_backend: TesseractTeacherPredictor | None = None,
    ) -> None:
        self.detector = detector or YoloOnnxTextDetector(
            model_path=detector_model_path,
            input_size=detector_input_size,
            confidence_threshold=detector_confidence_threshold,
            iou_threshold=detector_iou_threshold,
            min_box_size=detector_min_box_size,
            box_format=detector_box_format,
            padding_value=detector_padding_value,
            providers=detector_providers,
        )
        recognizer_config = dict(recognizer or {})
        recognizer_config.setdefault('default_direction', 'horizontal')
        recognizer_config.setdefault('default_text_type', default_text_type)
        recognizer_config.setdefault('min_text_height', 1.0)
        self.recognizer = recognizer_backend or TesseractTeacherPredictor(**recognizer_config)
        self.crop_padding = max(int(crop_padding), 0)
        self.min_crop_size = max(int(min_crop_size), 1)
        self.aspect_ratio_vertical_threshold = float(aspect_ratio_vertical_threshold)
        self.rotate_vertical_crops = bool(rotate_vertical_crops)
        self.default_text_type = TextType(default_text_type)
        self.crop_temp_root = Path(crop_temp_root) if crop_temp_root else (_REPO_ROOT / '.tmp-crops')

    def predict_page(self, asset: PageAsset):
        detections = self.detector.detect_path(asset.image_path)
        if not detections:
            return []

        page_image = self._open_image(asset.image_path)
        predictions: list[dict[str, object]] = []
        temp_dir_path: Path | None = None
        try:
            temp_root = self._ensure_temp_root()
            temp_dir_path = Path(tempfile.mkdtemp(prefix='manga-ocr-yolo-crops-', dir=temp_root))
            for index, detection in enumerate(detections):
                crop_bounds = self._expand_crop_bounds(detection.box, asset)
                if crop_bounds is None:
                    continue
                crop_image = self._crop_image(page_image, crop_bounds)
                direction = self._infer_direction(detection)
                rotated = direction == TextDirection.VERTICAL and self.rotate_vertical_crops
                if rotated:
                    crop_image = self._rotate_vertical_crop(crop_image)
                crop_path = temp_dir_path / f'crop-{index:04d}.png'
                self._save_crop(crop_image, crop_path)
                crop_asset = replace(
                    asset,
                    image_path=str(crop_path),
                    width=getattr(crop_image, 'size', (None, None))[0],
                    height=getattr(crop_image, 'size', (None, None))[1],
                )
                raw_predictions = self.recognizer.predict_page(crop_asset)
                predictions.extend(
                    self._project_predictions_to_page(
                        raw_predictions,
                        asset=asset,
                        crop_bounds=crop_bounds,
                        direction=direction,
                        detection=detection,
                        rotated=rotated,
                    )
                )
        finally:
            self._close_image(page_image)
            if temp_dir_path is not None:
                shutil.rmtree(temp_dir_path, ignore_errors=True)
        return predictions

    def _open_image(self, image_path: str | Path):
        if Image is None:
            raise RuntimeError('YOLO+Tesseract teacher requires Pillow for crop extraction.')
        return Image.open(image_path).convert('RGB')

    def _ensure_temp_root(self) -> Path:
        self.crop_temp_root.mkdir(parents=True, exist_ok=True)
        return self.crop_temp_root

    @staticmethod
    def _close_image(image: Any) -> None:
        close = getattr(image, 'close', None)
        if callable(close):
            close()

    @staticmethod
    def _crop_image(image: Any, crop_bounds: tuple[int, int, int, int]):
        return image.crop(crop_bounds)

    @staticmethod
    def _save_crop(image: Any, output_path: Path) -> None:
        image.save(output_path)

    @staticmethod
    def _rotate_vertical_crop(image: Any):
        if Image is None:
            raise RuntimeError('YOLO+Tesseract teacher requires Pillow for vertical crop rotation.')
        transpose = getattr(Image, 'Transpose', None)
        if transpose is not None:
            return image.transpose(transpose.ROTATE_270)
        return image.transpose(Image.ROTATE_270)

    def _project_predictions_to_page(
        self,
        raw_predictions: list[dict[str, object]],
        *,
        asset: PageAsset,
        crop_bounds: tuple[int, int, int, int],
        direction: TextDirection,
        detection: TextDetection,
        rotated: bool,
    ) -> list[dict[str, object]]:
        page_predictions: list[dict[str, object]] = []
        crop_left, crop_top, crop_right, crop_bottom = crop_bounds
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        for raw_prediction in raw_predictions:
            transcript = str(raw_prediction.get('transcript') or '').strip()
            if not transcript:
                continue
            polygon = self._map_polygon_to_page(
                raw_prediction.get('polygon') or [],
                crop_left=crop_left,
                crop_top=crop_top,
                crop_width=crop_width,
                crop_height=crop_height,
                rotated=rotated,
            )
            if not polygon:
                continue
            recognition_confidence = float(raw_prediction.get('recognition_confidence', 0.0))
            script_confidence = float(raw_prediction.get('script_confidence', recognition_confidence))
            teacher_agreement = min(detection.confidence, recognition_confidence, script_confidence)
            min_text_height = max(self._polygon_height(polygon), detection.text_height)
            page_predictions.append(
                {
                    'asset_id': asset.asset_id,
                    'series_id': asset.series_id,
                    'image_path': asset.image_path,
                    'domain': asset.domain.value,
                    'polygon': [{'x': point.x, 'y': point.y} for point in polygon],
                    'transcript': transcript,
                    'lang': str(raw_prediction.get('lang') or 'mixed'),
                    'direction': direction.value,
                    'text_type': str(raw_prediction.get('text_type') or self.default_text_type.value),
                    'detection_confidence': detection.confidence,
                    'recognition_confidence': recognition_confidence,
                    'script_confidence': script_confidence,
                    'teacher_agreement': teacher_agreement,
                    'min_text_height': min_text_height,
                }
            )
        return page_predictions

    def _expand_crop_bounds(
        self,
        box: tuple[float, float, float, float],
        asset: PageAsset,
    ) -> tuple[int, int, int, int] | None:
        image_width = int(asset.width or 0)
        image_height = int(asset.height or 0)
        if image_width <= 0 or image_height <= 0:
            raise RuntimeError(
                f'PageAsset {asset.asset_id} must include width and height to extract YOLO text crops.'
            )
        x1, y1, x2, y2 = box
        left = max(0, int(round(x1)) - self.crop_padding)
        top = max(0, int(round(y1)) - self.crop_padding)
        right = min(image_width, int(round(x2)) + self.crop_padding)
        bottom = min(image_height, int(round(y2)) + self.crop_padding)
        if (right - left) < self.min_crop_size or (bottom - top) < self.min_crop_size:
            return None
        return left, top, right, bottom

    def _infer_direction(self, detection: TextDetection) -> TextDirection:
        x1, y1, x2, y2 = detection.box
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        if height / width >= self.aspect_ratio_vertical_threshold:
            return TextDirection.VERTICAL
        return TextDirection.HORIZONTAL

    @staticmethod
    def _map_polygon_to_page(
        raw_polygon: list[dict[str, float]] | list[PolygonPoint] | list[tuple[float, float]],
        *,
        crop_left: int,
        crop_top: int,
        crop_width: int,
        crop_height: int,
        rotated: bool,
    ) -> list[PolygonPoint]:
        points: list[PolygonPoint] = []
        for raw_point in raw_polygon:
            if isinstance(raw_point, PolygonPoint):
                x, y = raw_point.x, raw_point.y
            elif isinstance(raw_point, dict):
                x, y = float(raw_point['x']), float(raw_point['y'])
            else:
                x, y = float(raw_point[0]), float(raw_point[1])
            if rotated:
                original_x = y
                original_y = float(crop_height) - x
                x, y = original_x, original_y
            points.append(PolygonPoint(float(crop_left) + x, float(crop_top) + y))
        return points

    @staticmethod
    def _polygon_height(points: list[PolygonPoint]) -> float:
        if not points:
            return 0.0
        ys = [point.y for point in points]
        return float(max(ys) - min(ys))
