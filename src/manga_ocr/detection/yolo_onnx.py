from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import PolygonPoint

try:  # Optional runtime dependency.
    import numpy as np
except ImportError:  # pragma: no cover - exercised through clean runtime errors
    np = None  # type: ignore[assignment]

try:  # Optional runtime dependency.
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised through clean runtime errors
    ort = None  # type: ignore[assignment]

try:  # Optional runtime dependency.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised through clean runtime errors
    Image = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class _LetterboxMetadata:
    original_width: int
    original_height: int
    square_size: int
    pad_left: int
    pad_top: int


@dataclass(slots=True)
class TextDetection:
    polygon: list[PolygonPoint]
    confidence: float
    class_id: int = 0
    label: str = 'text'

    @property
    def box(self) -> tuple[float, float, float, float]:
        xs = [point.x for point in self.polygon]
        ys = [point.y for point in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def text_height(self) -> float:
        _, top, _, bottom = self.box
        return float(bottom - top)

    def to_dict(self) -> dict[str, Any]:
        return {
            'polygon': [{'x': point.x, 'y': point.y} for point in self.polygon],
            'confidence': self.confidence,
            'class_id': self.class_id,
            'label': self.label,
            'box': self.box,
        }


class YoloOnnxTextDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 1280,
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.30,
        max_detections: int | None = None,
        min_box_size: float = 4.0,
        padding_value: int = 127,
        box_format: str = 'xyxy',
        providers: Sequence[str] | None = None,
        label_map: Mapping[int, str] | None = None,
        session: Any | None = None,
    ) -> None:
        if box_format not in {'xyxy', 'xywh'}:
            raise ValueError("box_format must be 'xyxy' or 'xywh'")
        self.model_path = self._resolve_model_path(model_path)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max_detections
        self.min_box_size = float(min_box_size)
        self.padding_value = int(padding_value)
        self.box_format = box_format
        self.providers = list(providers) if providers else None
        self.label_map = dict(label_map or {0: 'text'})
        self._session = session

    def detect_path(self, image_path: str | Path) -> list[TextDetection]:
        image_array, metadata = self._load_image(image_path)
        return self.detect_array(image_array, metadata)

    def detect_array(self, image_array: Any, metadata: _LetterboxMetadata | None = None) -> list[TextDetection]:
        self._require_runtime_dependencies(require_onnx=self._session is None)
        if metadata is None:
            height, width = int(image_array.shape[0]), int(image_array.shape[1])
            metadata = _LetterboxMetadata(
                original_width=width,
                original_height=height,
                square_size=max(width, height),
                pad_left=(max(width, height) - width) // 2,
                pad_top=(max(width, height) - height) // 2,
            )
        input_tensor, metadata = self._preprocess_image(image_array, metadata)
        outputs = self._run_session(input_tensor)
        rows = self._extract_rows(outputs)
        return self._decode_rows(rows, metadata)

    def _load_image(self, image_path: str | Path) -> tuple[Any, _LetterboxMetadata]:
        self._require_runtime_dependencies(require_onnx=False)
        image_file = Path(image_path)
        with Image.open(image_file) as image:  # type: ignore[union-attr]
            rgb_image = image.convert('RGB')
            width, height = rgb_image.size
            array = np.asarray(rgb_image, dtype=np.uint8)
        square_size = max(width, height)
        pad_left = (square_size - width) // 2
        pad_top = (square_size - height) // 2
        metadata = _LetterboxMetadata(
            original_width=width,
            original_height=height,
            square_size=square_size,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return array, metadata

    def _preprocess_image(self, image_array: Any, metadata: _LetterboxMetadata) -> tuple[Any, _LetterboxMetadata]:
        self._require_runtime_dependencies(require_onnx=False)
        square_size = metadata.square_size
        canvas = np.full((square_size, square_size, 3), self.padding_value, dtype=np.uint8)
        height, width = int(image_array.shape[0]), int(image_array.shape[1])
        left = metadata.pad_left
        top = metadata.pad_top
        canvas[top: top + height, left: left + width] = image_array
        resized = Image.fromarray(canvas).resize((self.input_size, self.input_size))  # type: ignore[union-attr]
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, metadata

    def _run_session(self, input_tensor: Any) -> list[Any]:
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        return list(session.run(None, {input_name: input_tensor}))

    def _get_session(self):
        if self._session is not None:
            return self._session
        self._require_runtime_dependencies()
        session_kwargs: dict[str, Any] = {}
        if self.providers:
            session_kwargs['providers'] = list(self.providers)
        self._session = ort.InferenceSession(str(self.model_path), **session_kwargs)  # type: ignore[union-attr]
        return self._session

    def _extract_rows(self, outputs: Sequence[Any]) -> list[list[float]]:
        if not outputs:
            return []
        primary = outputs[0]
        if np is not None:
            primary = np.asarray(primary)
            if primary.ndim == 3 and primary.shape[0] == 1:
                primary = primary[0]
            if primary.ndim == 2 and primary.shape[1] < 6 and primary.shape[0] >= 6:
                primary = primary.T
            if primary.ndim != 2:
                raise RuntimeError(f'Unsupported YOLO ONNX output shape: {tuple(primary.shape)}')
            return [[float(value) for value in row.tolist()] for row in primary]
        if isinstance(primary, Sequence) and primary and isinstance(primary[0], Sequence):
            first = primary[0]
            if len(primary) == 1 and isinstance(first, Sequence) and first and isinstance(first[0], Sequence):
                primary = first
            if primary and isinstance(primary[0], Sequence) and len(primary[0]) >= 6:
                return [[float(value) for value in row] for row in primary]
        raise RuntimeError('Unsupported YOLO ONNX output payload without numpy installed.')

    def _decode_rows(self, rows: Sequence[Sequence[float]], metadata: _LetterboxMetadata) -> list[TextDetection]:
        detections: list[TextDetection] = []
        scale = metadata.square_size / float(self.input_size)
        for row in rows:
            if len(row) < 5:
                continue
            score = float(row[4])
            if score < self.confidence_threshold:
                continue
            class_id = int(row[5]) if len(row) >= 6 else 0
            box = self._row_to_box(row)
            scaled_box = self._scale_box(box, scale, metadata)
            if scaled_box is None:
                continue
            x1, y1, x2, y2 = scaled_box
            polygon = [
                PolygonPoint(x1, y1),
                PolygonPoint(x2, y1),
                PolygonPoint(x2, y2),
                PolygonPoint(x1, y2),
            ]
            detections.append(
                TextDetection(
                    polygon=polygon,
                    confidence=score,
                    class_id=class_id,
                    label=self.label_map.get(class_id, str(class_id)),
                )
            )
        detections = self._nms(detections)
        if self.max_detections is not None:
            detections = detections[: max(self.max_detections, 0)]
        return detections

    def _row_to_box(self, row: Sequence[float]) -> tuple[float, float, float, float]:
        if self.box_format == 'xywh':
            center_x, center_y, width, height = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            half_width = width / 2.0
            half_height = height / 2.0
            return center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height
        return float(row[0]), float(row[1]), float(row[2]), float(row[3])

    def _scale_box(
        self,
        box: tuple[float, float, float, float],
        scale: float,
        metadata: _LetterboxMetadata,
    ) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = box
        x1 = (x1 * scale) - metadata.pad_left
        y1 = (y1 * scale) - metadata.pad_top
        x2 = (x2 * scale) - metadata.pad_left
        y2 = (y2 * scale) - metadata.pad_top

        x1 = max(0.0, min(float(metadata.original_width), x1))
        y1 = max(0.0, min(float(metadata.original_height), y1))
        x2 = max(0.0, min(float(metadata.original_width), x2))
        y2 = max(0.0, min(float(metadata.original_height), y2))

        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
            return None
        return x1, y1, x2, y2

    def _nms(self, detections: Sequence[TextDetection]) -> list[TextDetection]:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: list[TextDetection] = []
        for detection in ordered:
            if any(self._iou(detection.box, existing.box) >= self.iou_threshold for existing in kept):
                continue
            kept.append(detection)
        return kept

    @staticmethod
    def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        inter_width = max(0.0, x2 - x1)
        inter_height = max(0.0, y2 - y1)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        denominator = first_area + second_area - intersection
        if denominator <= 0:
            return 0.0
        return intersection / denominator

    def _resolve_model_path(self, model_path: str | Path) -> Path:
        candidate = Path(model_path)
        if candidate.is_absolute() or candidate.exists():
            return candidate
        repo_candidate = _REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate
        return candidate

    @staticmethod
    def _require_runtime_dependencies(require_onnx: bool = True) -> None:
        missing: list[str] = []
        if np is None:
            missing.append('numpy')
        if Image is None:
            missing.append('Pillow')
        if require_onnx and ort is None:
            missing.append('onnxruntime')
        if missing:
            packages = ', '.join(missing)
            raise RuntimeError(
                f'YOLO ONNX detection requires optional dependencies that are not installed: {packages}.'
            )
