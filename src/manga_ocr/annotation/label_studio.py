from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from ..domain import (
    DatasetManifestRecord,
    Domain,
    LabelSource,
    OcrLineLabel,
    PageAsset,
    PolygonPoint,
    TextDirection,
    TextType,
)
from ..manifests import write_jsonl
from ..splits import SeriesSplitStrategy

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABEL_CONFIG = (_REPO_ROOT / "configs/label_studio/manga_ocr_label_config.xml").read_text(
    encoding="utf-8"
)


def build_label_studio_tasks(
    page_assets: Sequence[PageAsset],
    predictions_by_asset: Mapping[str, Sequence[Any]] | None = None,
    image_base_url: str | None = None,
    local_files_document_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    predictions_by_asset = predictions_by_asset or {}

    for asset in page_assets:
        image_value = _build_image_value(
            asset.image_path,
            image_base_url=image_base_url,
            local_files_document_root=local_files_document_root,
        )

        task: dict[str, Any] = {
            "data": {
                "image": image_value,
                "image_path": asset.image_path,
                "asset_id": asset.asset_id,
                "series_id": asset.series_id,
                "chapter_id": asset.chapter_id,
                "page_index": asset.page_index,
                "domain": asset.domain.value,
                "width": asset.width,
                "height": asset.height,
            }
        }
        predictions = predictions_by_asset.get(asset.asset_id, [])
        if predictions:
            result = _build_prediction_results(predictions, asset.width, asset.height)
            task["predictions"] = [{"model_version": "teacher-bootstrap", "score": 0.0, "result": result}]
        tasks.append(task)
    return tasks


def write_label_studio_tasks(tasks: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(tasks), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_image_value(
    image_path: str,
    image_base_url: str | None,
    local_files_document_root: str | Path | None,
) -> str:
    if image_base_url and local_files_document_root:
        raise ValueError("Use either 'image_base_url' or 'local_files_document_root', not both.")
    if image_base_url:
        return f"{image_base_url.rstrip('/')}/{Path(image_path).name}"
    if local_files_document_root:
        document_root = Path(local_files_document_root).expanduser().resolve()
        candidate = Path(image_path)
        image_file = candidate if candidate.is_absolute() else (document_root / candidate)
        image_file = image_file.resolve()
        try:
            relative_path = image_file.relative_to(document_root)
        except ValueError as exc:
            raise ValueError(
                f"Image path '{image_path}' is outside Label Studio local files root '{document_root}'."
            ) from exc
        return f"/data/local-files/?d={quote(relative_path.as_posix(), safe='/')}"
    return image_path


def convert_label_studio_export(
    export_payload: Sequence[dict[str, Any]],
    split_strategy: SeriesSplitStrategy | None = None,
) -> list[DatasetManifestRecord]:
    split_strategy = split_strategy or SeriesSplitStrategy()
    records: list[DatasetManifestRecord] = []

    for task in export_payload:
        data = task.get("data", {})
        annotations = task.get("annotations") or []
        predictions = task.get("predictions") or []

        source_items = annotations if annotations else predictions
        source_label = LabelSource.HUMAN if annotations else LabelSource.TEACHER
        for source_item in source_items:
            result_items = source_item.get("result", [])
            grouped = _group_results(result_items)
            for group in grouped.values():
                polygon_item = next((item for item in group if item.get("type") == "polygonlabels"), None)
                if polygon_item is None:
                    continue
                width = polygon_item.get("original_width") or data.get("width")
                height = polygon_item.get("original_height") or data.get("height")
                polygon = _decode_polygon_points(polygon_item["value"].get("points", []), width, height)
                transcript = _extract_text(group)
                if not transcript:
                    continue
                lang = _extract_choice(group, "language", default="mixed")
                direction = TextDirection(_extract_choice(group, "direction", default="horizontal"))
                text_type = TextType(
                    next(iter(polygon_item.get("value", {}).get("polygonlabels", ["unknown"])), "unknown")
                )
                records.append(
                    DatasetManifestRecord(
                        image_path=str(data.get("image_path", "")),
                        series_id=str(data.get("series_id", "")),
                        domain=Domain(str(data.get("domain", "manga"))),
                        tile_id=data.get("tile_id"),
                        polygon=polygon,
                        transcript=transcript,
                        lang=lang,
                        direction=direction,
                        text_type=text_type,
                        split=split_strategy.assign(str(data.get("series_id", ""))),
                        confidence=source_item.get("score"),
                        source=source_label,
                    )
                )

    return records


def export_label_studio_jsonl(
    export_payload: Sequence[dict[str, Any]],
    output_path: str | Path,
    split_strategy: SeriesSplitStrategy | None = None,
) -> list[DatasetManifestRecord]:
    records = convert_label_studio_export(export_payload, split_strategy=split_strategy)
    write_jsonl((record.to_dict() for record in records), output_path)
    return records


def _build_prediction_results(
    predictions: Sequence[Any], width: int | None, height: int | None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    image_width = width or 1
    image_height = height or 1
    for index, prediction in enumerate(predictions):
        polygon, transcript, lang, direction, text_type = _coerce_prediction(prediction)
        region_id = f"pred-{index}"
        results.append(
            {
                "id": region_id,
                "from_name": "text_region",
                "to_name": "image",
                "type": "polygonlabels",
                "original_width": image_width,
                "original_height": image_height,
                "value": {
                    "points": [
                        [round((point.x / image_width) * 100, 4), round((point.y / image_height) * 100, 4)]
                        for point in polygon
                    ],
                    "polygonlabels": [text_type.value],
                },
            }
        )
        results.append(
            {
                "id": f"{region_id}-text",
                "parentID": region_id,
                "from_name": "transcript",
                "to_name": "image",
                "type": "textarea",
                "value": {"text": [transcript]},
            }
        )
        results.append(
            {
                "id": f"{region_id}-language",
                "parentID": region_id,
                "from_name": "language",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": [lang]},
            }
        )
        results.append(
            {
                "id": f"{region_id}-direction",
                "parentID": region_id,
                "from_name": "direction",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": [direction.value]},
            }
        )
    return results


def _coerce_prediction(
    prediction: Any,
) -> tuple[list[PolygonPoint], str, str, TextDirection, TextType]:
    if isinstance(prediction, DatasetManifestRecord):
        return prediction.polygon, prediction.transcript, prediction.lang, prediction.direction, prediction.text_type
    if isinstance(prediction, OcrLineLabel):
        return (
            prediction.polygon,
            prediction.transcript,
            prediction.lang,
            prediction.direction,
            prediction.text_type,
        )
    polygon = [PolygonPoint(float(point["x"]), float(point["y"])) for point in prediction["polygon"]]
    transcript = str(prediction["transcript"])
    lang = str(prediction["lang"])
    direction = TextDirection(str(prediction["direction"]))
    text_type = TextType(str(prediction["text_type"]))
    return polygon, transcript, lang, direction, text_type


def _group_results(result_items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in result_items:
        key = str(item.get("id") or item.get("parentID") or "orphan")
        key = str(item.get("parentID") or key)
        groups.setdefault(key, []).append(item)
    return groups


def _decode_polygon_points(
    raw_points: Sequence[Sequence[float]],
    width: int | None,
    height: int | None,
) -> list[PolygonPoint]:
    if width and height:
        return [
            PolygonPoint(x=(point[0] / 100.0) * width, y=(point[1] / 100.0) * height)
            for point in raw_points
        ]
    return [PolygonPoint(x=point[0], y=point[1]) for point in raw_points]


def _extract_text(group: Sequence[dict[str, Any]]) -> str:
    for item in group:
        if item.get("type") == "textarea":
            texts = item.get("value", {}).get("text", [])
            if texts:
                return str(texts[0]).strip()
    return ""


def _extract_choice(group: Sequence[dict[str, Any]], from_name: str, default: str) -> str:
    for item in group:
        if item.get("type") == "choices" and item.get("from_name") == from_name:
            choices = item.get("value", {}).get("choices", [])
            if choices:
                return str(choices[0])
    return default
