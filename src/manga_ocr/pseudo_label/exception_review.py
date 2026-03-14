from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..annotation.label_studio import build_label_studio_tasks
from ..domain import PageAsset
from ..manifests import write_jsonl
from .filtering import PseudoLabelThresholds, TeacherPrediction, evaluate_prediction


@dataclass(slots=True)
class ExceptionReviewPolicy:
    audit_rate: float = 0.02
    min_audit_pages: int = 10
    max_audit_pages: int | None = None
    audit_seed: str = 'manga-ocr-audit-v1'
    review_empty_pages: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.audit_rate <= 1.0:
            raise ValueError('audit_rate must be between 0.0 and 1.0')
        if self.min_audit_pages < 0:
            raise ValueError('min_audit_pages must be >= 0')
        if self.max_audit_pages is not None and self.max_audit_pages < 0:
            raise ValueError('max_audit_pages must be >= 0 when provided')


@dataclass(slots=True)
class ExceptionReviewDecision:
    asset: PageAsset
    route: str
    review_score: float
    reasons: list[str]
    accepted_predictions: int
    rejected_predictions: int
    preannotated_lines: int
    sample_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'asset_id': self.asset.asset_id,
            'image_path': self.asset.image_path,
            'series_id': self.asset.series_id,
            'chapter_id': self.asset.chapter_id,
            'page_index': self.asset.page_index,
            'domain': self.asset.domain.value,
            'route': self.route,
            'review_score': self.review_score,
            'reasons': list(self.reasons),
            'accepted_predictions': self.accepted_predictions,
            'rejected_predictions': self.rejected_predictions,
            'preannotated_lines': self.preannotated_lines,
            'sample_score': self.sample_score,
        }


def build_review_by_exception_tasks(
    page_assets: Sequence[PageAsset],
    predictions: Sequence[TeacherPrediction],
    thresholds: PseudoLabelThresholds | None = None,
    policy: ExceptionReviewPolicy | None = None,
    image_base_url: str | None = None,
    local_files_document_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[ExceptionReviewDecision]]:
    policy = policy or ExceptionReviewPolicy()
    predictions = list(predictions)
    grouped_predictions = _group_predictions_by_asset(predictions)

    decisions: list[ExceptionReviewDecision] = []
    accepted_only: list[ExceptionReviewDecision] = []
    for asset in page_assets:
        asset_predictions = grouped_predictions.get(asset.asset_id, [])
        decision = _decision_for_asset(asset, asset_predictions, thresholds=thresholds, policy=policy)
        if decision is None:
            continue
        decisions.append(decision)
        if decision.route == 'auto_accept':
            accepted_only.append(decision)

    audit_count = _target_audit_pages(len(accepted_only), policy)
    if audit_count:
        sampled = sorted(
            accepted_only,
            key=lambda item: (_sample_score(item.asset.asset_id, policy.audit_seed), item.asset.asset_id),
        )[:audit_count]
        for decision in sampled:
            decision.route = 'audit'
            decision.sample_score = _sample_score(decision.asset.asset_id, policy.audit_seed)
            decision.reasons = ['audit_sample']

    selected_decisions = [decision for decision in decisions if decision.route in {'review', 'audit'}]
    selected_decisions.sort(
        key=lambda item: (
            0 if item.route == 'review' else 1,
            -item.review_score if item.route == 'review' else item.sample_score if item.sample_score is not None else 1.0,
            item.asset.series_id,
            item.asset.chapter_id,
            item.asset.page_index,
        )
    )

    selected_assets = [decision.asset for decision in selected_decisions]
    predictions_by_asset = {
        decision.asset.asset_id: [prediction.to_line_label() for prediction in grouped_predictions.get(decision.asset.asset_id, [])]
        for decision in selected_decisions
    }
    tasks = build_label_studio_tasks(
        selected_assets,
        predictions_by_asset=predictions_by_asset,
        image_base_url=image_base_url,
        local_files_document_root=local_files_document_root,
    )
    for task, decision in zip(tasks, selected_decisions, strict=False):
        task['data'].update(
            {
                'review_route': decision.route,
                'review_score': decision.review_score,
                'triage_reasons': list(decision.reasons),
                'accepted_predictions': decision.accepted_predictions,
                'rejected_predictions': decision.rejected_predictions,
                'preannotated_lines': decision.preannotated_lines,
                'audit_sample_score': decision.sample_score,
            }
        )
    return tasks, decisions


def summarize_exception_review(decisions: Sequence[ExceptionReviewDecision]) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    preannotated_lines = 0
    accepted_predictions = 0
    rejected_predictions = 0
    for decision in decisions:
        route_counts.update([decision.route])
        if decision.route == 'review':
            reason_counts.update(decision.reasons)
        preannotated_lines += decision.preannotated_lines
        accepted_predictions += decision.accepted_predictions
        rejected_predictions += decision.rejected_predictions

    accepted_only_pages = route_counts.get('auto_accept', 0) + route_counts.get('audit', 0)
    audit_pages = route_counts.get('audit', 0)
    effective_audit_rate = 0.0
    if accepted_only_pages:
        effective_audit_rate = audit_pages / float(accepted_only_pages)

    return {
        'pages_considered': len(decisions),
        'auto_accept_pages': route_counts.get('auto_accept', 0),
        'review_pages': route_counts.get('review', 0),
        'audit_pages': audit_pages,
        'selected_review_tasks': route_counts.get('review', 0) + audit_pages,
        'accepted_predictions': accepted_predictions,
        'rejected_predictions': rejected_predictions,
        'preannotated_lines': preannotated_lines,
        'review_reason_counts': dict(reason_counts),
        'effective_audit_rate': effective_audit_rate,
    }


def write_exception_review_decisions(
    decisions: Sequence[ExceptionReviewDecision],
    output_path: str | Path,
) -> None:
    write_jsonl((decision.to_dict() for decision in decisions), output_path)


def _decision_for_asset(
    asset: PageAsset,
    predictions: Sequence[TeacherPrediction],
    thresholds: PseudoLabelThresholds | None,
    policy: ExceptionReviewPolicy,
) -> ExceptionReviewDecision | None:
    if not predictions:
        if not policy.review_empty_pages:
            return None
        return ExceptionReviewDecision(
            asset=asset,
            route='review',
            review_score=2.0,
            reasons=['no_teacher_predictions'],
            accepted_predictions=0,
            rejected_predictions=0,
            preannotated_lines=0,
            sample_score=None,
        )

    prediction_decisions = [evaluate_prediction(prediction, thresholds=thresholds) for prediction in predictions]
    accepted_predictions = sum(1 for decision in prediction_decisions if decision.accepted)
    rejected_predictions = len(prediction_decisions) - accepted_predictions
    reasons = sorted({reason for decision in prediction_decisions for reason in decision.reasons})
    review_score = max(decision.review_score for decision in prediction_decisions)
    route = 'review' if rejected_predictions else 'auto_accept'
    return ExceptionReviewDecision(
        asset=asset,
        route=route,
        review_score=review_score,
        reasons=reasons,
        accepted_predictions=accepted_predictions,
        rejected_predictions=rejected_predictions,
        preannotated_lines=len(predictions),
        sample_score=None,
    )


def _group_predictions_by_asset(predictions: Sequence[TeacherPrediction]) -> dict[str, list[TeacherPrediction]]:
    grouped: dict[str, list[TeacherPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.asset_id, []).append(prediction)
    return grouped


def _sample_score(asset_id: str, audit_seed: str) -> float:
    digest = hashlib.sha256(f'{audit_seed}:{asset_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(2**64 - 1)


def _target_audit_pages(candidate_count: int, policy: ExceptionReviewPolicy) -> int:
    if candidate_count <= 0:
        return 0
    audit_count = math.ceil(candidate_count * policy.audit_rate)
    audit_count = max(audit_count, policy.min_audit_pages)
    if policy.max_audit_pages is not None:
        audit_count = min(audit_count, policy.max_audit_pages)
    return min(audit_count, candidate_count)
