from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..ingest import ChapterDescriptor, ConsumetMangahereAdapter, IngestionPipeline
from ..manifests import write_page_assets_manifest
from ..pseudo_label import (
    ExceptionReviewPolicy,
    PseudoLabelThresholds,
    load_teacher_predictor,
    run_teacher_cycle,
)
from ..storage import CatalogDatabase

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_YOLO_TEACHER_CONFIG = _REPO_ROOT / 'configs' / 'teacher_yolo26_tesseract.json'
_DEFAULT_PREDICTOR = 'manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor'


@dataclass(slots=True)
class ColabDatasetWorkspace:
    root: Path
    data_dir: Path
    raw_dir: Path
    manifests_dir: Path
    annotation_tasks_dir: Path
    artifacts_dir: Path
    configs_dir: Path
    db_path: Path
    page_assets_manifest: Path
    teacher_predictions_manifest: Path
    silver_manifest: Path
    rejected_predictions_manifest: Path
    review_queue_path: Path
    review_decisions_path: Path
    summary_path: Path
    archive_path: Path
    predictor_config_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> 'ColabDatasetWorkspace':
        root_path = Path(root).expanduser().resolve()
        return cls(
            root=root_path,
            data_dir=root_path / 'data',
            raw_dir=root_path / 'data' / 'raw',
            manifests_dir=root_path / 'manifests',
            annotation_tasks_dir=root_path / 'annotation' / 'tasks',
            artifacts_dir=root_path / 'artifacts',
            configs_dir=root_path / 'configs',
            db_path=root_path / 'data' / 'catalog.sqlite',
            page_assets_manifest=root_path / 'manifests' / 'page_assets.jsonl',
            teacher_predictions_manifest=root_path / 'manifests' / 'teacher_predictions.jsonl',
            silver_manifest=root_path / 'manifests' / 'silver_manifest.jsonl',
            rejected_predictions_manifest=root_path / 'manifests' / 'rejected_predictions.jsonl',
            review_queue_path=root_path / 'annotation' / 'tasks' / 'review_queue.json',
            review_decisions_path=root_path / 'manifests' / 'review_decisions.jsonl',
            summary_path=root_path / 'artifacts' / 'dataset_summary.json',
            archive_path=root_path / 'artifacts' / 'dataset_bundle.zip',
            predictor_config_path=root_path / 'configs' / 'teacher_yolo26_tesseract.json',
        )

    def initialize(self) -> None:
        for directory in (
            self.root,
            self.data_dir,
            self.raw_dir,
            self.manifests_dir,
            self.annotation_tasks_dir,
            self.artifacts_dir,
            self.configs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class IngestionBatchSummary:
    chapter_id: str
    assets: int
    errors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'chapter_id': self.chapter_id,
            'assets': self.assets,
            'errors': self.errors,
        }


def prepare_dataset_bundle(
    *,
    workspace_root: str | Path,
    manga_id: str | None = None,
    latest_page: int | None = None,
    latest_limit: int | None = None,
    chapter_limit: int = 1,
    all_chapters: bool = False,
    predictor_import_path: str = _DEFAULT_PREDICTOR,
    predictor_config_path: str | Path | None = None,
    detector_model_path: str | Path | None = None,
    run_teacher: bool = True,
    create_archive: bool = True,
    review_strategy: str = 'exception-audit',
    review_page_limit: int = 100,
    audit_rate: float = 0.02,
    min_audit_pages: int = 10,
    max_audit_pages: int | None = None,
    review_empty_pages: bool = True,
    min_detection_confidence: float | None = None,
    min_recognition_confidence: float | None = None,
    min_script_confidence: float | None = None,
    min_teacher_agreement: float | None = None,
    min_text_height: float | None = None,
    continue_on_error: bool = True,
    base_url: str = 'https://api-consumet-org-extension.vercel.app',
    referer: str = 'https://mangahere.com/',
    timeout_s: float = 30.0,
    metadata_timeout_s: float | None = None,
    image_timeout_s: float | None = 120.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    source_id: str = 'consumet-mangahere',
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    if not manga_id and latest_page is None:
        raise ValueError('Provide either manga_id or latest_page.')
    workspace = ColabDatasetWorkspace.from_root(workspace_root)
    workspace.initialize()

    catalog = CatalogDatabase(workspace.db_path)
    catalog.initialize()
    adapter = ConsumetMangahereAdapter(
        base_url=base_url,
        referer=referer,
        timeout_s=timeout_s,
        metadata_timeout_s=metadata_timeout_s,
        image_timeout_s=image_timeout_s,
        source_id=source_id,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    pipeline = IngestionPipeline(adapter=adapter, catalog=catalog, storage_root=workspace.data_dir)

    chapter_descriptors = _resolve_chapter_descriptors(
        adapter,
        manga_id=manga_id,
        latest_page=latest_page,
        latest_limit=latest_limit,
        chapter_limit=chapter_limit,
        all_chapters=all_chapters,
    )
    ingested_assets = []
    ingestion_summaries: list[IngestionBatchSummary] = []
    ingestion_errors: list[dict[str, Any]] = []
    for descriptor in chapter_descriptors:
        report = pipeline.ingest_chapter_with_report(
            descriptor.chapter_id,
            continue_on_error=continue_on_error,
        )
        ingested_assets.extend(report.assets)
        ingestion_summaries.append(
            IngestionBatchSummary(
                chapter_id=descriptor.chapter_id,
                assets=len(report.assets),
                errors=len(report.errors),
            )
        )
        ingestion_errors.extend(error.to_dict() for error in report.errors)

    write_page_assets_manifest(ingested_assets, workspace.page_assets_manifest)

    teacher_summary: dict[str, Any] | None = None
    effective_predictor_config: Path | None = None
    if run_teacher:
        effective_predictor_config = _resolve_predictor_config(
            workspace=workspace,
            predictor_config_path=predictor_config_path,
            detector_model_path=detector_model_path,
        )
        predictor = load_teacher_predictor(predictor_import_path, str(effective_predictor_config))
        thresholds = PseudoLabelThresholds(
            **{
                key: value
                for key, value in {
                    'min_detection_confidence': min_detection_confidence,
                    'min_recognition_confidence': min_recognition_confidence,
                    'min_script_confidence': min_script_confidence,
                    'min_teacher_agreement': min_teacher_agreement,
                    'min_text_height': min_text_height,
                }.items()
                if value is not None
            }
        )
        exception_policy = ExceptionReviewPolicy(
            audit_rate=audit_rate,
            min_audit_pages=min_audit_pages,
            max_audit_pages=max_audit_pages,
            review_empty_pages=review_empty_pages,
        )
        cycle = run_teacher_cycle(
            predictor,
            ingested_assets,
            predictions_output_path=workspace.teacher_predictions_manifest,
            silver_output_path=workspace.silver_manifest,
            review_output_path=workspace.review_queue_path,
            thresholds=thresholds,
            continue_on_error=continue_on_error,
            page_limit=review_page_limit,
            rejected_output_path=workspace.rejected_predictions_manifest,
            review_decisions_output_path=workspace.review_decisions_path,
            review_strategy=review_strategy,
            exception_policy=exception_policy,
        )
        teacher_summary = cycle.summary()

    archive_destination = Path(archive_path).expanduser().resolve() if archive_path else workspace.archive_path

    summary = {
        'workspace_root': str(workspace.root),
        'source': {
            'manga_id': manga_id,
            'latest_page': latest_page,
            'latest_limit': latest_limit,
            'chapter_limit': chapter_limit,
            'all_chapters': all_chapters,
        },
        'chapters_requested': len(chapter_descriptors),
        'ingestion': {
            'chapters': [item.to_dict() for item in ingestion_summaries],
            'total_assets': len(ingested_assets),
            'total_errors': len(ingestion_errors),
            'errors': ingestion_errors,
        },
        'artifacts': {
            'catalog_db': str(workspace.db_path),
            'raw_dir': str(workspace.raw_dir),
            'page_assets_manifest': str(workspace.page_assets_manifest),
            'teacher_predictions_manifest': str(workspace.teacher_predictions_manifest) if run_teacher else None,
            'silver_manifest': str(workspace.silver_manifest) if run_teacher else None,
            'rejected_predictions_manifest': str(workspace.rejected_predictions_manifest) if run_teacher else None,
            'review_queue_path': str(workspace.review_queue_path) if run_teacher else None,
            'review_decisions_path': str(workspace.review_decisions_path) if run_teacher else None,
            'predictor_config': str(effective_predictor_config) if effective_predictor_config else None,
            'archive_path': str(archive_destination) if create_archive else None,
        },
        'teacher_cycle': teacher_summary,
    }
    workspace.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if create_archive:
        _build_dataset_archive(
            workspace=workspace,
            output_path=archive_destination,
            include_teacher_outputs=run_teacher,
            predictor_config_path=effective_predictor_config,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='prepare-colab-dataset')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manga-id')
    source.add_argument('--latest-page', type=int)
    parser.add_argument('--workspace-root', default='/content/manga-ocr-workdir')
    parser.add_argument('--latest-limit', type=int)
    parser.add_argument('--chapter-limit', type=int, default=1)
    parser.add_argument('--all-chapters', action='store_true')
    parser.add_argument('--predictor', default=_DEFAULT_PREDICTOR)
    parser.add_argument('--predictor-config')
    parser.add_argument('--detector-model-path')
    parser.add_argument('--skip-teacher', action='store_true')
    parser.add_argument('--skip-archive', action='store_true')
    parser.add_argument('--archive-path')
    parser.add_argument('--review-strategy', choices=['topk', 'exception-audit'], default='exception-audit')
    parser.add_argument('--review-page-limit', type=int, default=100)
    parser.add_argument('--audit-rate', type=float, default=0.02)
    parser.add_argument('--min-audit-pages', type=int, default=10)
    parser.add_argument('--max-audit-pages', type=int)
    parser.add_argument('--skip-empty-page-review', dest='review_empty_pages', action='store_false')
    parser.set_defaults(review_empty_pages=True)
    parser.add_argument('--min-detection-confidence', type=float)
    parser.add_argument('--min-recognition-confidence', type=float)
    parser.add_argument('--min-script-confidence', type=float)
    parser.add_argument('--min-teacher-agreement', type=float)
    parser.add_argument('--min-text-height', type=float)
    parser.add_argument('--continue-on-error', action='store_true', default=True)
    parser.add_argument('--base-url', default='https://api-consumet-org-extension.vercel.app')
    parser.add_argument('--referer', default='https://mangahere.com/')
    parser.add_argument('--timeout-s', type=float, default=30.0)
    parser.add_argument('--metadata-timeout-s', type=float)
    parser.add_argument('--image-timeout-s', type=float, default=120.0)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retry-backoff-s', type=float, default=1.0)
    parser.add_argument('--source-id', default='consumet-mangahere')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = prepare_dataset_bundle(
        workspace_root=args.workspace_root,
        manga_id=args.manga_id,
        latest_page=args.latest_page,
        latest_limit=args.latest_limit,
        chapter_limit=args.chapter_limit,
        all_chapters=args.all_chapters,
        predictor_import_path=args.predictor,
        predictor_config_path=args.predictor_config,
        detector_model_path=args.detector_model_path,
        run_teacher=not args.skip_teacher,
        create_archive=not args.skip_archive,
        review_strategy=args.review_strategy,
        review_page_limit=args.review_page_limit,
        audit_rate=args.audit_rate,
        min_audit_pages=args.min_audit_pages,
        max_audit_pages=args.max_audit_pages,
        review_empty_pages=args.review_empty_pages,
        min_detection_confidence=args.min_detection_confidence,
        min_recognition_confidence=args.min_recognition_confidence,
        min_script_confidence=args.min_script_confidence,
        min_teacher_agreement=args.min_teacher_agreement,
        min_text_height=args.min_text_height,
        continue_on_error=args.continue_on_error,
        base_url=args.base_url,
        referer=args.referer,
        timeout_s=args.timeout_s,
        metadata_timeout_s=args.metadata_timeout_s,
        image_timeout_s=args.image_timeout_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
        source_id=args.source_id,
        archive_path=args.archive_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _resolve_chapter_descriptors(
    adapter: ConsumetMangahereAdapter,
    *,
    manga_id: str | None,
    latest_page: int | None,
    latest_limit: int | None,
    chapter_limit: int,
    all_chapters: bool,
) -> list[ChapterDescriptor]:
    if manga_id:
        limit = None if all_chapters else max(chapter_limit, 1)
        return adapter.get_manga_chapters(manga_id, limit=limit)
    listing = adapter.list_latest_chapters(page=max(int(latest_page or 1), 1), limit=latest_limit)
    return list(listing.chapters)


def _resolve_predictor_config(
    *,
    workspace: ColabDatasetWorkspace,
    predictor_config_path: str | Path | None,
    detector_model_path: str | Path | None,
) -> Path:
    if predictor_config_path:
        candidate = Path(predictor_config_path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        repo_candidate = _REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate
        raise FileNotFoundError(f'Could not find predictor config: {predictor_config_path}')
    if not detector_model_path:
        raise ValueError('detector_model_path is required when predictor_config is not provided.')
    payload = json.loads(_DEFAULT_YOLO_TEACHER_CONFIG.read_text(encoding='utf-8'))
    payload['detector_model_path'] = str(Path(detector_model_path).expanduser())
    workspace.predictor_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return workspace.predictor_config_path


def _archive_member_path(workspace_root: Path, path: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return f'external/{path.name}'


def _build_dataset_archive(
    *,
    workspace: ColabDatasetWorkspace,
    output_path: Path,
    include_teacher_outputs: bool,
    predictor_config_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items: list[Path] = [
        workspace.db_path,
        workspace.page_assets_manifest,
        workspace.raw_dir,
        workspace.summary_path,
    ]
    if predictor_config_path is not None:
        items.append(predictor_config_path)
    if include_teacher_outputs:
        items.extend(
            [
                workspace.teacher_predictions_manifest,
                workspace.silver_manifest,
                workspace.rejected_predictions_manifest,
                workspace.review_queue_path,
                workspace.review_decisions_path,
            ]
        )
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            if not item.exists():
                continue
            if item.is_dir():
                for child in item.rglob('*'):
                    if child.is_file():
                        archive.write(child, _archive_member_path(workspace.root, child))
                continue
            archive.write(item, _archive_member_path(workspace.root, item))


if __name__ == '__main__':  # pragma: no cover
    main()
