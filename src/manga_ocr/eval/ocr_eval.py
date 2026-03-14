from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..domain import DatasetSplit, Domain, PolygonPoint, TextDirection, TextType
from ..manifests import read_jsonl
from ..pseudo_label import TeacherPrediction


@dataclass(slots=True)
class GoldLineRecord:
    image_path: str
    series_id: str
    domain: Domain
    polygon: list[PolygonPoint]
    transcript: str
    lang: str
    direction: TextDirection
    text_type: TextType
    split: DatasetSplit


@dataclass(slots=True)
class _MatchedPair:
    gold: GoldLineRecord
    prediction: TeacherPrediction
    iou: float
    char_edits: int
    gold_char_count: int
    word_edits: int
    gold_word_count: int


@dataclass(slots=True)
class _BucketAggregate:
    gold_lines: int = 0
    matched_lines: int = 0
    gold_char_count: int = 0
    matched_gold_char_count: int = 0
    matched_char_edits: int = 0
    unmatched_gold_char_count: int = 0
    gold_word_count: int = 0
    matched_gold_word_count: int = 0
    matched_word_edits: int = 0
    unmatched_gold_word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        matched_cer = _safe_divide(self.matched_char_edits, self.matched_gold_char_count)
        end_to_end_cer = _safe_divide(
            self.matched_char_edits + self.unmatched_gold_char_count,
            self.gold_char_count,
        )
        matched_wer = _safe_divide(self.matched_word_edits, self.matched_gold_word_count)
        end_to_end_wer = _safe_divide(
            self.matched_word_edits + self.unmatched_gold_word_count,
            self.gold_word_count,
        )
        return {
            "gold_lines": self.gold_lines,
            "matched_lines": self.matched_lines,
            "unmatched_gold_lines": self.gold_lines - self.matched_lines,
            "detection_recall": _safe_divide(self.matched_lines, self.gold_lines),
            "matched_cer": matched_cer,
            "end_to_end_cer": end_to_end_cer,
            "matched_wer": matched_wer,
            "end_to_end_wer": end_to_end_wer,
        }


def load_gold_manifest_records(input_path: str | Path) -> list[GoldLineRecord]:
    records = read_jsonl(input_path)
    return [_gold_line_from_dict(record) for record in records]


def evaluate_predictions(
    gold_records: Sequence[GoldLineRecord],
    predictions: Sequence[TeacherPrediction],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    grouped_gold = _group_by_image_path(gold_records)
    grouped_predictions = _group_by_image_path(predictions)
    image_paths = sorted(set(grouped_gold) | set(grouped_predictions))

    matched_pairs: list[_MatchedPair] = []
    unmatched_gold: list[GoldLineRecord] = []
    unmatched_predictions: list[TeacherPrediction] = []

    for image_path in image_paths:
        image_gold = grouped_gold.get(image_path, [])
        image_predictions = grouped_predictions.get(image_path, [])
        image_matches, image_unmatched_gold, image_unmatched_predictions = _match_image_lines(
            image_gold,
            image_predictions,
            iou_threshold=iou_threshold,
        )
        matched_pairs.extend(image_matches)
        unmatched_gold.extend(image_unmatched_gold)
        unmatched_predictions.extend(image_unmatched_predictions)

    total_gold_lines = len(gold_records)
    total_prediction_lines = len(predictions)
    total_gold_chars = sum(_char_count(record.transcript) for record in gold_records)
    total_gold_words = sum(_word_count(record.transcript) for record in gold_records)
    matched_gold_chars = sum(pair.gold_char_count for pair in matched_pairs)
    matched_gold_words = sum(pair.gold_word_count for pair in matched_pairs)
    matched_char_edits = sum(pair.char_edits for pair in matched_pairs)
    matched_word_edits = sum(pair.word_edits for pair in matched_pairs)
    unmatched_gold_chars = sum(_char_count(record.transcript) for record in unmatched_gold)
    unmatched_gold_words = sum(_word_count(record.transcript) for record in unmatched_gold)

    overall = {
        "gold_lines": total_gold_lines,
        "prediction_lines": total_prediction_lines,
        "matched_lines": len(matched_pairs),
        "unmatched_gold_lines": len(unmatched_gold),
        "unmatched_prediction_lines": len(unmatched_predictions),
        "detection_recall": _safe_divide(len(matched_pairs), total_gold_lines),
        "detection_precision": _safe_divide(len(matched_pairs), total_prediction_lines),
        "matched_cer": _safe_divide(matched_char_edits, matched_gold_chars),
        "end_to_end_cer": _safe_divide(matched_char_edits + unmatched_gold_chars, total_gold_chars),
        "matched_wer": _safe_divide(matched_word_edits, matched_gold_words),
        "end_to_end_wer": _safe_divide(matched_word_edits + unmatched_gold_words, total_gold_words),
        "iou_threshold": iou_threshold,
    }

    slices = {
        "domain": _build_slice_metrics(gold_records, matched_pairs, unmatched_gold, lambda record: record.domain.value),
        "direction": _build_slice_metrics(
            gold_records,
            matched_pairs,
            unmatched_gold,
            lambda record: record.direction.value,
        ),
        "text_type": _build_slice_metrics(
            gold_records,
            matched_pairs,
            unmatched_gold,
            lambda record: record.text_type.value,
        ),
        "split": _build_slice_metrics(gold_records, matched_pairs, unmatched_gold, lambda record: record.split.value),
    }

    return {
        "overall": overall,
        "slices": slices,
    }


def _group_by_image_path(records: Iterable[GoldLineRecord] | Iterable[TeacherPrediction]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[str(record.image_path)].append(record)
    return dict(grouped)


def _match_image_lines(
    gold_records: Sequence[GoldLineRecord],
    predictions: Sequence[TeacherPrediction],
    iou_threshold: float,
) -> tuple[list[_MatchedPair], list[GoldLineRecord], list[TeacherPrediction]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for gold_index, gold_record in enumerate(gold_records):
        gold_bbox = _polygon_bbox(gold_record.polygon)
        for prediction_index, prediction in enumerate(predictions):
            prediction_bbox = _polygon_bbox(prediction.polygon)
            iou = _bbox_iou(gold_bbox, prediction_bbox)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gold_index, prediction_index))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
    used_gold: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[_MatchedPair] = []

    for iou, gold_index, prediction_index in candidate_pairs:
        if gold_index in used_gold or prediction_index in used_predictions:
            continue
        gold_record = gold_records[gold_index]
        prediction = predictions[prediction_index]
        gold_text = _normalize_text(gold_record.transcript)
        prediction_text = _normalize_text(prediction.transcript)
        matches.append(
            _MatchedPair(
                gold=gold_record,
                prediction=prediction,
                iou=iou,
                char_edits=_levenshtein(list(gold_text), list(prediction_text)),
                gold_char_count=_char_count(gold_text),
                word_edits=_levenshtein(gold_text.split(), prediction_text.split()),
                gold_word_count=_word_count(gold_text),
            )
        )
        used_gold.add(gold_index)
        used_predictions.add(prediction_index)

    unmatched_gold = [record for index, record in enumerate(gold_records) if index not in used_gold]
    unmatched_predictions = [record for index, record in enumerate(predictions) if index not in used_predictions]
    return matches, unmatched_gold, unmatched_predictions


def _build_slice_metrics(
    gold_records: Sequence[GoldLineRecord],
    matched_pairs: Sequence[_MatchedPair],
    unmatched_gold: Sequence[GoldLineRecord],
    key_fn,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, _BucketAggregate] = defaultdict(_BucketAggregate)
    matched_by_gold_id = {id(pair.gold): pair for pair in matched_pairs}
    unmatched_gold_ids = {id(record) for record in unmatched_gold}

    for gold_record in gold_records:
        bucket = buckets[str(key_fn(gold_record))]
        bucket.gold_lines += 1
        bucket.gold_char_count += _char_count(gold_record.transcript)
        bucket.gold_word_count += _word_count(gold_record.transcript)

        matched_pair = matched_by_gold_id.get(id(gold_record))
        if matched_pair is not None:
            bucket.matched_lines += 1
            bucket.matched_gold_char_count += matched_pair.gold_char_count
            bucket.matched_char_edits += matched_pair.char_edits
            bucket.matched_gold_word_count += matched_pair.gold_word_count
            bucket.matched_word_edits += matched_pair.word_edits
        elif id(gold_record) in unmatched_gold_ids:
            bucket.unmatched_gold_char_count += _char_count(gold_record.transcript)
            bucket.unmatched_gold_word_count += _word_count(gold_record.transcript)

    return {key: bucket.to_dict() for key, bucket in sorted(buckets.items())}


def _gold_line_from_dict(data: dict[str, Any]) -> GoldLineRecord:
    return GoldLineRecord(
        image_path=str(data["image_path"]),
        series_id=str(data["series_id"]),
        domain=Domain(str(data.get("domain", Domain.MANGA.value))),
        polygon=_parse_polygon(data.get("polygon")),
        transcript=str(data.get("transcript") or ""),
        lang=str(data.get("lang") or "mixed"),
        direction=TextDirection(str(data.get("direction") or TextDirection.HORIZONTAL.value)),
        text_type=TextType(str(data.get("text_type") or TextType.UNKNOWN.value)),
        split=DatasetSplit(str(data.get("split") or DatasetSplit.TRAIN.value)),
    )


def _parse_polygon(raw_polygon: Any) -> list[PolygonPoint]:
    points: list[PolygonPoint] = []
    for point in raw_polygon or []:
        points.append(PolygonPoint(float(point["x"]), float(point["y"])))
    return points


def _polygon_bbox(polygon: Sequence[PolygonPoint]) -> tuple[float, float, float, float]:
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _char_count(text: str) -> int:
    return len(_normalize_text(text))


def _word_count(text: str) -> int:
    normalized = _normalize_text(text)
    return len(normalized.split()) if normalized else 0


def _levenshtein(source: Sequence[str], target: Sequence[str]) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            insertions = previous_row[target_index] + 1
            deletions = current_row[target_index - 1] + 1
            substitutions = previous_row[target_index - 1] + (0 if source_item == target_item else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)
