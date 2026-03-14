from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..annotation.label_studio import write_label_studio_tasks
from ..domain import DatasetManifestRecord, PageAsset
from ..manifests import write_dataset_manifest
from .exception_review import (
    ExceptionReviewPolicy,
    build_review_by_exception_tasks,
    summarize_exception_review,
    write_exception_review_decisions,
)
from .filtering import FilterDecision, PseudoLabelThresholds, TeacherPrediction
from .io import build_review_tasks, build_silver_manifest, summarize_predictions, write_filter_decisions
from .teacher import TeacherAssetFailure, TeacherPredictor, run_teacher_predictions, write_teacher_predictions


@dataclass(slots=True)
class TeacherCycleArtifacts:
    predictions: list[TeacherPrediction]
    failures: list[TeacherAssetFailure]
    silver_records: list[DatasetManifestRecord]
    rejected: list[FilterDecision]
    review_tasks: list[dict[str, Any]]
    review_decisions: list[Any]
    thresholds: PseudoLabelThresholds | None = None
    review_summary: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        summary = summarize_predictions(self.predictions, thresholds=self.thresholds)
        summary.update(
            {
                "failed_assets": len(self.failures),
                "review_tasks": len(self.review_tasks),
                "review_decisions": len(self.review_decisions),
            }
        )
        if self.review_summary:
            summary.update(self.review_summary)
        return summary


def run_teacher_cycle(
    predictor: TeacherPredictor,
    page_assets: Sequence[PageAsset],
    predictions_output_path: str | Path,
    silver_output_path: str | Path,
    review_output_path: str | Path,
    thresholds: PseudoLabelThresholds | None = None,
    continue_on_error: bool = True,
    page_limit: int = 100,
    image_base_url: str | None = None,
    local_files_document_root: str | Path | None = None,
    rejected_output_path: str | Path | None = None,
    review_decisions_output_path: str | Path | None = None,
    review_strategy: str = 'topk',
    exception_policy: ExceptionReviewPolicy | None = None,
) -> TeacherCycleArtifacts:
    predictions, failures = run_teacher_predictions(
        predictor,
        page_assets,
        continue_on_error=continue_on_error,
    )
    write_teacher_predictions(predictions, predictions_output_path)

    silver_records, rejected = build_silver_manifest(predictions, thresholds=thresholds)
    write_dataset_manifest(silver_records, silver_output_path)
    if rejected_output_path:
        write_filter_decisions(rejected, rejected_output_path)

    review_summary: dict[str, Any] | None = None
    if review_strategy == 'exception-audit':
        review_tasks, review_decisions = build_review_by_exception_tasks(
            page_assets,
            predictions,
            thresholds=thresholds,
            policy=exception_policy,
            image_base_url=image_base_url,
            local_files_document_root=local_files_document_root,
        )
        review_summary = summarize_exception_review(review_decisions)
    else:
        review_tasks, review_decisions = build_review_tasks(
            page_assets,
            predictions,
            page_limit=page_limit,
            thresholds=thresholds,
            image_base_url=image_base_url,
            local_files_document_root=local_files_document_root,
        )
    write_label_studio_tasks(review_tasks, review_output_path)
    if review_decisions_output_path:
        if review_strategy == 'exception-audit':
            write_exception_review_decisions(review_decisions, review_decisions_output_path)
        else:
            write_filter_decisions(review_decisions, review_decisions_output_path)

    return TeacherCycleArtifacts(
        predictions=predictions,
        failures=failures,
        silver_records=silver_records,
        rejected=rejected,
        review_tasks=review_tasks,
        review_decisions=review_decisions,
        thresholds=thresholds,
        review_summary=review_summary,
    )
