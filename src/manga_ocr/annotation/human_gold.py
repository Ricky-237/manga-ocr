from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..domain import DatasetSplit, PageAsset
from ..manifests import write_jsonl
from ..pseudo_label.filtering import PseudoLabelThresholds, TeacherPrediction, evaluate_prediction
from ..pseudo_label.io import group_predictions_by_asset
from ..splits import SeriesSplitStrategy
from .label_studio import build_label_studio_tasks


@dataclass(slots=True)
class HumanGoldSelection:
    asset: PageAsset
    split: DatasetSplit
    review_mode: str
    review_priority: float
    priority_reasons: list[str]
    preannotated_lines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset.asset_id,
            "image_path": self.asset.image_path,
            "series_id": self.asset.series_id,
            "chapter_id": self.asset.chapter_id,
            "page_index": self.asset.page_index,
            "domain": self.asset.domain.value,
            "split": self.split.value,
            "review_mode": self.review_mode,
            "review_priority": self.review_priority,
            "priority_reasons": list(self.priority_reasons),
            "preannotated_lines": self.preannotated_lines,
        }


def select_human_gold_pages(
    page_assets: Sequence[PageAsset],
    predictions: Sequence[TeacherPrediction] | None = None,
    page_limit: int = 2500,
    split_strategy: SeriesSplitStrategy | None = None,
    thresholds: PseudoLabelThresholds | None = None,
) -> list[HumanGoldSelection]:
    split_strategy = split_strategy or SeriesSplitStrategy()
    predictions = list(predictions or [])
    predictions_by_asset = group_predictions_by_asset(predictions)

    candidates: list[HumanGoldSelection] = []
    for asset in page_assets:
        asset_predictions = predictions_by_asset.get(asset.asset_id, [])
        split = split_strategy.assign(asset.series_id)
        review_mode = "double_review" if split == DatasetSplit.TEST else "single_review"
        review_priority = 0.0
        priority_reasons: list[str] = []
        if asset_predictions:
            decisions = [evaluate_prediction(prediction, thresholds=thresholds) for prediction in asset_predictions]
            review_priority = max(decision.review_score for decision in decisions)
            reason_set = {reason for decision in decisions for reason in decision.reasons}
            priority_reasons = sorted(reason_set)
        candidates.append(
            HumanGoldSelection(
                asset=asset,
                split=split,
                review_mode=review_mode,
                review_priority=review_priority,
                priority_reasons=priority_reasons,
                preannotated_lines=len(asset_predictions),
            )
        )

    candidates.sort(
        key=lambda selection: (
            -selection.review_priority,
            selection.asset.series_id,
            selection.asset.chapter_id,
            selection.asset.page_index,
        )
    )
    return _apply_split_quota(candidates, page_limit=page_limit, split_strategy=split_strategy)


def build_human_gold_tasks(
    selections: Sequence[HumanGoldSelection],
    predictions: Sequence[TeacherPrediction] | None = None,
    image_base_url: str | None = None,
    local_files_document_root: str | None = None,
) -> list[dict[str, Any]]:
    predictions = list(predictions or [])
    selected_asset_ids = {selection.asset.asset_id for selection in selections}
    selected_assets = [selection.asset for selection in selections]
    predictions_by_asset = {
        asset_id: [prediction.to_line_label() for prediction in prediction_list]
        for asset_id, prediction_list in group_predictions_by_asset(predictions).items()
        if asset_id in selected_asset_ids
    }
    tasks = build_label_studio_tasks(
        selected_assets,
        predictions_by_asset=predictions_by_asset,
        image_base_url=image_base_url,
        local_files_document_root=local_files_document_root,
    )
    for rank, (task, selection) in enumerate(zip(tasks, selections, strict=False), start=1):
        task["data"].update(
            {
                "target_split": selection.split.value,
                "review_mode": selection.review_mode,
                "review_priority": selection.review_priority,
                "priority_reasons": list(selection.priority_reasons),
                "preannotated_lines": selection.preannotated_lines,
                "gold_batch_rank": rank,
            }
        )
    return tasks


def write_human_gold_manifest(selections: Sequence[HumanGoldSelection], output_path: str) -> None:
    write_jsonl((selection.to_dict() for selection in selections), output_path)


def summarize_human_gold_batch(selections: Sequence[HumanGoldSelection]) -> dict[str, Any]:
    split_counts: dict[str, int] = defaultdict(int)
    review_mode_counts: dict[str, int] = defaultdict(int)
    preannotated_pages = 0
    for selection in selections:
        split_counts[selection.split.value] += 1
        review_mode_counts[selection.review_mode] += 1
        if selection.preannotated_lines:
            preannotated_pages += 1
    return {
        "selected_pages": len(selections),
        "preannotated_pages": preannotated_pages,
        "split_counts": dict(sorted(split_counts.items())),
        "review_mode_counts": dict(sorted(review_mode_counts.items())),
    }


def _apply_split_quota(
    candidates: Sequence[HumanGoldSelection],
    page_limit: int,
    split_strategy: SeriesSplitStrategy,
) -> list[HumanGoldSelection]:
    if page_limit <= 0 or not candidates:
        return []

    limit = min(page_limit, len(candidates))
    by_split: dict[DatasetSplit, list[HumanGoldSelection]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VAL: [],
        DatasetSplit.TEST: [],
    }
    for candidate in candidates:
        by_split[candidate.split].append(candidate)

    target_counts = _ideal_split_counts(limit, split_strategy)
    available_splits = [split for split, values in by_split.items() if values]
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        if split not in available_splits:
            target_counts[split] = 0

    minimum_counts = {
        split: 1 if split in available_splits and limit >= len(available_splits) else 0
        for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
    }
    for split, minimum in minimum_counts.items():
        target_counts[split] = max(target_counts[split], minimum)

    while sum(target_counts.values()) > limit:
        reducible_splits = [
            split
            for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
            if target_counts[split] > minimum_counts[split]
        ]
        if not reducible_splits:
            break
        split_to_reduce = max(
            reducible_splits,
            key=lambda split: (target_counts[split], split.value),
        )
        target_counts[split_to_reduce] -= 1

    selected: list[HumanGoldSelection] = []
    leftovers: list[HumanGoldSelection] = []
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        split_candidates = by_split[split]
        take = min(target_counts[split], len(split_candidates))
        selected.extend(split_candidates[:take])
        leftovers.extend(split_candidates[take:])

    if len(selected) < limit:
        leftovers.sort(
            key=lambda selection: (
                -selection.review_priority,
                selection.asset.series_id,
                selection.asset.chapter_id,
                selection.asset.page_index,
            )
        )
        selected.extend(leftovers[: limit - len(selected)])

    selected.sort(
        key=lambda selection: (
            -selection.review_priority,
            selection.asset.series_id,
            selection.asset.chapter_id,
            selection.asset.page_index,
        )
    )
    return selected


def _ideal_split_counts(limit: int, split_strategy: SeriesSplitStrategy) -> dict[DatasetSplit, int]:
    ratios = {
        DatasetSplit.TRAIN: split_strategy.train_ratio,
        DatasetSplit.VAL: split_strategy.val_ratio,
        DatasetSplit.TEST: split_strategy.test_ratio,
    }
    exact = {split: limit * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in exact.items()}
    assigned = sum(counts.values())
    remainders = sorted(
        ((exact[split] - counts[split], split) for split in ratios),
        key=lambda item: item[0],
        reverse=True,
    )
    for _, split in remainders[: limit - assigned]:
        counts[split] += 1
    return counts
