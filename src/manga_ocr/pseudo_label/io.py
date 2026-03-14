from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..annotation.label_studio import build_label_studio_tasks
from ..domain import DatasetManifestRecord, LabelSource, PageAsset, PolygonPoint, TextDirection, TextType
from ..manifests import read_jsonl, write_dataset_manifest, write_jsonl
from ..splits import SeriesSplitStrategy
from .filtering import FilterDecision, PseudoLabelThresholds, TeacherPrediction, evaluate_prediction, filter_for_silver


def read_prediction_records(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return read_jsonl(path)

    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return [dict(item) for item in payload["predictions"]]
    if isinstance(payload, dict):
        return [dict(payload)]
    raise ValueError(f"Unsupported prediction payload in {path}")


def teacher_predictions_from_records(
    records: Sequence[Mapping[str, Any]],
    page_assets_by_id: Mapping[str, PageAsset] | None = None,
) -> list[TeacherPrediction]:
    return [teacher_prediction_from_dict(record, page_assets_by_id=page_assets_by_id) for record in records]


def load_teacher_predictions(
    input_path: str | Path,
    page_assets_by_id: Mapping[str, PageAsset] | None = None,
) -> list[TeacherPrediction]:
    records = read_prediction_records(input_path)
    return teacher_predictions_from_records(records, page_assets_by_id=page_assets_by_id)


def teacher_prediction_from_dict(
    data: Mapping[str, Any],
    page_assets_by_id: Mapping[str, PageAsset] | None = None,
) -> TeacherPrediction:
    asset_id = str(data.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError("Teacher prediction requires 'asset_id'.")

    asset = page_assets_by_id.get(asset_id) if page_assets_by_id else None
    series_id = _coalesce_str(data.get("series_id"), asset.series_id if asset else None, field_name="series_id")
    image_path = _coalesce_str(data.get("image_path"), asset.image_path if asset else None, field_name="image_path")
    domain_value = _coalesce_str(
        data.get("domain"),
        asset.domain.value if asset else None,
        field_name="domain",
        default="manga",
    )

    confidence_fallback = float(data.get("confidence", 0.0))
    transcript = str(data.get("transcript") or "").strip()
    if not transcript:
        raise ValueError(f"Teacher prediction '{asset_id}' requires a non-empty transcript.")

    return TeacherPrediction(
        asset_id=asset_id,
        series_id=series_id,
        image_path=image_path,
        domain=asset.domain if asset else _parse_domain(domain_value),
        polygon=_parse_polygon(data.get("polygon")),
        transcript=transcript,
        lang=str(data.get("lang") or "mixed"),
        direction=TextDirection(str(data.get("direction") or "horizontal")),
        text_type=TextType(str(data.get("text_type") or "unknown")),
        detection_confidence=float(data.get("detection_confidence", confidence_fallback)),
        recognition_confidence=float(data.get("recognition_confidence", confidence_fallback)),
        script_confidence=float(data.get("script_confidence", confidence_fallback)),
        teacher_agreement=float(data.get("teacher_agreement", confidence_fallback)),
        min_text_height=float(data["min_text_height"]) if data.get("min_text_height") is not None else None,
    )


def group_predictions_by_asset(
    predictions: Sequence[TeacherPrediction],
) -> dict[str, list[TeacherPrediction]]:
    grouped: dict[str, list[TeacherPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.asset_id].append(prediction)
    return dict(grouped)


def build_silver_manifest(
    predictions: Sequence[TeacherPrediction],
    split_strategy: SeriesSplitStrategy | None = None,
    thresholds: PseudoLabelThresholds | None = None,
) -> tuple[list[DatasetManifestRecord], list[FilterDecision]]:
    split_strategy = split_strategy or SeriesSplitStrategy()
    accepted, rejected = filter_for_silver(predictions, thresholds=thresholds)
    records = [
        DatasetManifestRecord(
            image_path=prediction.image_path,
            series_id=prediction.series_id,
            domain=prediction.domain,
            tile_id=None,
            polygon=prediction.polygon,
            transcript=prediction.transcript,
            lang=prediction.lang,
            direction=prediction.direction,
            text_type=prediction.text_type,
            split=split_strategy.assign(prediction.series_id),
            confidence=prediction.confidence_score,
            source=LabelSource.TEACHER,
        )
        for prediction in accepted
    ]
    return records, rejected


def export_silver_manifest(
    predictions: Sequence[TeacherPrediction],
    output_path: str | Path,
    split_strategy: SeriesSplitStrategy | None = None,
    thresholds: PseudoLabelThresholds | None = None,
) -> tuple[list[DatasetManifestRecord], list[FilterDecision]]:
    records, rejected = build_silver_manifest(
        predictions,
        split_strategy=split_strategy,
        thresholds=thresholds,
    )
    write_dataset_manifest(records, output_path)
    return records, rejected


def build_review_tasks(
    page_assets: Sequence[PageAsset],
    predictions: Sequence[TeacherPrediction],
    page_limit: int = 100,
    thresholds: PseudoLabelThresholds | None = None,
    image_base_url: str | None = None,
    local_files_document_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[FilterDecision]]:
    asset_lookup = {asset.asset_id: asset for asset in page_assets}
    grouped_predictions = group_predictions_by_asset(predictions)
    decisions = [evaluate_prediction(prediction, thresholds=thresholds) for prediction in predictions]
    decisions.sort(key=lambda item: item.review_score, reverse=True)

    selected_asset_ids: list[str] = []
    seen_asset_ids: set[str] = set()
    for decision in decisions:
        asset_id = decision.prediction.asset_id
        if asset_id not in asset_lookup or asset_id in seen_asset_ids:
            continue
        seen_asset_ids.add(asset_id)
        selected_asset_ids.append(asset_id)
        if len(selected_asset_ids) >= page_limit:
            break

    selected_assets = [asset_lookup[asset_id] for asset_id in selected_asset_ids]
    predictions_by_asset = {
        asset_id: [prediction.to_line_label() for prediction in grouped_predictions.get(asset_id, [])]
        for asset_id in selected_asset_ids
    }
    tasks = build_label_studio_tasks(
        selected_assets,
        predictions_by_asset=predictions_by_asset,
        image_base_url=image_base_url,
        local_files_document_root=local_files_document_root,
    )
    selected_decisions = [decision for decision in decisions if decision.prediction.asset_id in seen_asset_ids]
    return tasks, selected_decisions


def summarize_predictions(
    predictions: Sequence[TeacherPrediction],
    thresholds: PseudoLabelThresholds | None = None,
) -> dict[str, Any]:
    accepted, rejected = filter_for_silver(predictions, thresholds=thresholds)
    reason_counter: Counter[str] = Counter()
    for decision in rejected:
        reason_counter.update(decision.reasons)
    return {
        "input_predictions": len(predictions),
        "accepted_silver": len(accepted),
        "review_candidates": len(rejected),
        "rejection_reasons": dict(reason_counter),
    }


def write_filter_decisions(decisions: Sequence[FilterDecision], output_path: str | Path) -> None:
    write_jsonl((decision.to_dict() for decision in decisions), output_path)


def _parse_polygon(raw_polygon: Any) -> list[PolygonPoint]:
    if not isinstance(raw_polygon, Sequence) or not raw_polygon:
        raise ValueError("Teacher prediction requires a non-empty 'polygon'.")

    points: list[PolygonPoint] = []
    for raw_point in raw_polygon:
        if isinstance(raw_point, Mapping):
            x = raw_point.get("x")
            y = raw_point.get("y")
        elif isinstance(raw_point, Sequence) and len(raw_point) >= 2:
            x, y = raw_point[0], raw_point[1]
        else:
            raise ValueError(f"Unsupported polygon point format: {raw_point!r}")
        points.append(PolygonPoint(float(x), float(y)))
    return points


def _parse_domain(value: str):
    from ..domain import Domain

    return Domain(str(value))


def _coalesce_str(*values: Any, field_name: str, default: str | None = None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    if default is not None:
        return default
    raise ValueError(f"Teacher prediction requires '{field_name}'.")
