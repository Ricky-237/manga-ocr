from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from .annotation import (
    build_human_gold_tasks,
    build_label_studio_tasks,
    convert_label_studio_export,
    select_human_gold_pages,
    summarize_human_gold_batch,
    write_human_gold_manifest,
    write_label_studio_tasks,
)
from .domain import PageAsset
from .eval import evaluate_predictions, load_gold_manifest_records
from .ingest import ChapterApiAdapter, ChapterDescriptor, ConsumetMangahereAdapter, IngestionPipeline
from .manifests import write_dataset_manifest, write_page_assets_manifest
from .pseudo_label import (
    ExceptionReviewPolicy,
    build_review_by_exception_tasks,
    PseudoLabelThresholds,
    build_review_tasks,
    build_silver_manifest,
    load_teacher_predictor,
    read_prediction_records,
    run_teacher_cycle,
    run_teacher_predictions,
    select_page_assets,
    summarize_exception_review,
    summarize_predictions,
    teacher_predictions_from_records,
    write_exception_review_decisions,
    write_filter_decisions,
    write_teacher_predictions,
)
from .storage import CatalogDatabase
from .train.pipeline import OcrTrainingPipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="manga-ocr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser("init-db", help="Initialize the SQLite catalog.")
    init_db_parser.add_argument("--db", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a chapter through a ChapterApiAdapter.")
    ingest_parser.add_argument("--adapter", required=True, help="Import path like package.module:ClassName")
    ingest_parser.add_argument("--adapter-config", help="Optional JSON config passed to the adapter constructor.")
    ingest_parser.add_argument("--db", required=True)
    ingest_parser.add_argument("--storage-root", required=True)
    ingest_parser.add_argument("--chapter-id", required=True)

    ingest_latest_parser = subparsers.add_parser(
        "ingest-latest",
        help="Ingest the latest chapters listed on a Consumet Mangahere page.",
    )
    ingest_latest_parser.add_argument("--db", required=True)
    ingest_latest_parser.add_argument("--storage-root", required=True)
    ingest_latest_parser.add_argument("--page", type=int, default=1)
    ingest_latest_parser.add_argument("--limit", type=int)
    _add_consumet_args(ingest_latest_parser)

    ingest_manga_parser = subparsers.add_parser(
        "ingest-manga",
        help="Ingest recent or all chapters for a single Consumet Mangahere manga.",
    )
    ingest_manga_parser.add_argument("--db", required=True)
    ingest_manga_parser.add_argument("--storage-root", required=True)
    ingest_manga_parser.add_argument("--manga-id", required=True)
    ingest_manga_parser.add_argument("--chapter-limit", type=int, default=1)
    ingest_manga_parser.add_argument("--all-chapters", action="store_true")
    _add_consumet_args(ingest_manga_parser)

    assets_parser = subparsers.add_parser("export-assets-manifest", help="Export page assets to JSONL.")
    assets_parser.add_argument("--db", required=True)
    assets_parser.add_argument("--out", required=True)

    ls_tasks_parser = subparsers.add_parser("build-label-studio-tasks", help="Export Label Studio tasks.")
    ls_tasks_parser.add_argument("--db", required=True)
    ls_tasks_parser.add_argument("--out", required=True)
    _add_label_studio_image_source_args(ls_tasks_parser)

    ls_export_parser = subparsers.add_parser(
        "convert-label-studio-export", help="Convert Label Studio export to dataset_manifest.jsonl."
    )
    ls_export_parser.add_argument("--input", required=True)
    ls_export_parser.add_argument("--out", required=True)

    human_gold_parser = subparsers.add_parser(
        "prepare-human-gold",
        help="Prepare a human-reviewed gold batch with stable splits and optional teacher pre-annotations.",
    )
    human_gold_parser.add_argument("--db", required=True)
    human_gold_parser.add_argument("--tasks-out", required=True)
    human_gold_parser.add_argument("--manifest-out", required=True)
    human_gold_parser.add_argument("--predictions")
    human_gold_parser.add_argument("--page-limit", type=int, default=2500)
    _add_label_studio_image_source_args(human_gold_parser)
    _add_asset_selection_args(human_gold_parser)
    _add_threshold_args(human_gold_parser)

    run_teacher_parser = subparsers.add_parser(
        "run-teacher",
        help="Run a pluggable teacher predictor over catalog assets and export canonical teacher_predictions.jsonl.",
    )
    run_teacher_parser.add_argument("--db", required=True)
    run_teacher_parser.add_argument("--out", required=True)
    run_teacher_parser.add_argument("--predictor", required=True, help="Import path like package.module:ClassName")
    run_teacher_parser.add_argument("--predictor-config", help="Optional JSON config passed to the predictor constructor.")
    _add_asset_selection_args(run_teacher_parser)
    _add_continue_on_error_args(run_teacher_parser, default=True)

    run_teacher_cycle_parser = subparsers.add_parser(
        "run-teacher-cycle",
        help="Run teacher predictions, build silver_manifest, and export a review queue in one command.",
    )
    run_teacher_cycle_parser.add_argument("--db", required=True)
    run_teacher_cycle_parser.add_argument("--predictor", required=True, help="Import path like package.module:ClassName")
    run_teacher_cycle_parser.add_argument(
        "--predictor-config", help="Optional JSON config passed to the predictor constructor."
    )
    run_teacher_cycle_parser.add_argument("--predictions-out", required=True)
    run_teacher_cycle_parser.add_argument("--silver-out", required=True)
    run_teacher_cycle_parser.add_argument("--review-out", required=True)
    run_teacher_cycle_parser.add_argument("--rejected-out")
    run_teacher_cycle_parser.add_argument("--review-decisions-out")
    run_teacher_cycle_parser.add_argument("--page-limit", type=int, default=100)
    _add_review_strategy_args(run_teacher_cycle_parser)
    _add_label_studio_image_source_args(run_teacher_cycle_parser)
    _add_asset_selection_args(run_teacher_cycle_parser)
    _add_threshold_args(run_teacher_cycle_parser)
    _add_continue_on_error_args(run_teacher_cycle_parser, default=True)

    silver_parser = subparsers.add_parser(
        "build-silver-manifest",
        help="Filter teacher predictions into a canonical silver dataset manifest.",
    )
    silver_parser.add_argument(
        "--input", required=True, help="Teacher predictions as JSON, JSONL, or {predictions:[...]}"
    )
    silver_parser.add_argument("--out", required=True)
    silver_parser.add_argument("--db", help="Optional catalog DB used to fill missing asset fields from asset_id.")
    silver_parser.add_argument("--rejected-out", help="Optional JSONL for rejected predictions and reasons.")
    _add_threshold_args(silver_parser)

    review_parser = subparsers.add_parser(
        "build-review-queue",
        help="Prioritize hard teacher predictions and export Label Studio review tasks.",
    )
    review_parser.add_argument("--db", required=True)
    review_parser.add_argument("--predictions", required=True)
    review_parser.add_argument("--out", required=True)
    review_parser.add_argument("--page-limit", type=int, default=100)
    _add_review_strategy_args(review_parser)
    _add_label_studio_image_source_args(review_parser)
    review_parser.add_argument("--decisions-out", help="Optional JSONL for ranked review decisions.")
    _add_threshold_args(review_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate-predictions",
        help="Evaluate OCR predictions against a reviewed gold dataset manifest.",
    )
    evaluate_parser.add_argument("--gold", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--db", help="Optional catalog DB used to fill missing prediction asset fields.")
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--out")

    plan_parser = subparsers.add_parser("plan-training", help="Inspect the staged training plan.")
    plan_parser.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "init-db":
        _init_db(args.db)
    elif args.command == "ingest":
        _ingest(args.adapter, args.adapter_config, args.db, args.storage_root, args.chapter_id)
    elif args.command == "ingest-latest":
        _ingest_latest(args)
    elif args.command == "ingest-manga":
        _ingest_manga(args)
    elif args.command == "export-assets-manifest":
        _export_assets_manifest(args.db, args.out)
    elif args.command == "build-label-studio-tasks":
        _build_label_studio_tasks(args.db, args.out, args.image_base_url, args.local_files_document_root)
    elif args.command == "convert-label-studio-export":
        _convert_label_studio_export(args.input, args.out)
    elif args.command == "prepare-human-gold":
        _prepare_human_gold(args)
    elif args.command == "run-teacher":
        _run_teacher(args)
    elif args.command == "run-teacher-cycle":
        _run_teacher_cycle_command(args)
    elif args.command == "build-silver-manifest":
        _build_silver_manifest(args)
    elif args.command == "build-review-queue":
        _build_review_queue(args)
    elif args.command == "evaluate-predictions":
        _evaluate_predictions_command(args)
    elif args.command == "plan-training":
        _plan_training(args.config)


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-detection-confidence", type=float)
    parser.add_argument("--min-recognition-confidence", type=float)
    parser.add_argument("--min-script-confidence", type=float)
    parser.add_argument("--min-teacher-agreement", type=float)
    parser.add_argument("--min-text-height", type=float)


def _add_label_studio_image_source_args(parser: argparse.ArgumentParser) -> None:
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image-base-url")
    source_group.add_argument("--local-files-document-root")


def _add_review_strategy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-strategy",
        choices=["topk", "exception-audit"],
        default="topk",
        help="Use top-k hard pages or a review-by-exception flow with audit sampling.",
    )
    parser.add_argument("--audit-rate", type=float, default=0.02)
    parser.add_argument("--min-audit-pages", type=int, default=10)
    parser.add_argument("--max-audit-pages", type=int)
    parser.add_argument("--audit-seed", default="manga-ocr-audit-v1")
    parser.add_argument(
        "--skip-empty-page-review",
        dest="review_empty_pages",
        action="store_false",
        help="Do not send pages with zero teacher predictions to review-by-exception.",
    )
    parser.set_defaults(review_empty_pages=True)


def _add_asset_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-id")
    parser.add_argument("--series-id")
    parser.add_argument("--chapter-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-duplicates", action="store_true")


def _add_continue_on_error_args(parser: argparse.ArgumentParser, default: bool = True) -> None:
    error_mode = parser.add_mutually_exclusive_group()
    error_mode.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="Keep processing remaining items when one item fails.",
    )
    error_mode.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately on the first error.",
    )
    parser.set_defaults(continue_on_error=default)


def _add_consumet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="https://api-consumet-org-extension.vercel.app")
    parser.add_argument("--referer", default="https://mangahere.com/")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--metadata-timeout-s", type=float)
    parser.add_argument("--image-timeout-s", type=float)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--source-id", default="consumet-mangahere")
    _add_continue_on_error_args(parser, default=True)


def _build_thresholds(args: argparse.Namespace) -> PseudoLabelThresholds:
    values = {
        "min_detection_confidence": args.min_detection_confidence,
        "min_recognition_confidence": args.min_recognition_confidence,
        "min_script_confidence": args.min_script_confidence,
        "min_teacher_agreement": args.min_teacher_agreement,
        "min_text_height": args.min_text_height,
    }
    filtered = {key: value for key, value in values.items() if value is not None}
    return PseudoLabelThresholds(**filtered)


def _build_exception_review_policy(args: argparse.Namespace) -> ExceptionReviewPolicy:
    return ExceptionReviewPolicy(
        audit_rate=args.audit_rate,
        min_audit_pages=args.min_audit_pages,
        max_audit_pages=args.max_audit_pages,
        audit_seed=args.audit_seed,
        review_empty_pages=args.review_empty_pages,
    )


def _resolve_input_path(path_value: str | Path, must_exist: bool = False) -> Path:
    candidate = Path(path_value)
    candidates = [candidate]
    if candidate.is_absolute():
        if candidate.exists() or not must_exist:
            return candidate
    else:
        if candidate.exists():
            return candidate
        repo_candidate = _REPO_ROOT / candidate
        candidates.append(repo_candidate)
        if repo_candidate.exists():
            return repo_candidate
    if must_exist:
        tried = ", ".join(str(path.resolve()) if not path.is_absolute() else str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find input '{path_value}'. Tried: {tried}")
    return candidate


def _init_db(db_path: str) -> None:
    catalog = CatalogDatabase(_resolve_input_path(db_path))
    catalog.initialize()
    print(f"Initialized catalog at {db_path}")


def _ingest(
    adapter_import_path: str,
    adapter_config_path: str | None,
    db_path: str,
    storage_root: str,
    chapter_id: str,
) -> None:
    adapter = _load_adapter(adapter_import_path, adapter_config_path)
    pipeline = IngestionPipeline(adapter=adapter, catalog=CatalogDatabase(_resolve_input_path(db_path)), storage_root=storage_root)
    results = pipeline.ingest_chapter(chapter_id)
    print(json.dumps([asset.to_dict() for asset in results], ensure_ascii=False, indent=2))


def _ingest_latest(args: argparse.Namespace) -> None:
    adapter = _build_consumet_adapter(args)
    listing = adapter.list_latest_chapters(page=args.page, limit=args.limit)
    results = _ingest_descriptors(
        adapter,
        listing.chapters,
        args.db,
        args.storage_root,
        continue_on_error=args.continue_on_error,
    )
    payload = {
        "page": args.page,
        "requested_chapters": len(listing.chapters),
        "next_cursor": listing.next_cursor,
        **_summarize_ingest_results(results),
        "chapters": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _ingest_manga(args: argparse.Namespace) -> None:
    adapter = _build_consumet_adapter(args)
    limit = None if args.all_chapters else max(args.chapter_limit, 1)
    chapters = adapter.get_manga_chapters(args.manga_id, limit=limit)
    results = _ingest_descriptors(
        adapter,
        chapters,
        args.db,
        args.storage_root,
        continue_on_error=args.continue_on_error,
    )
    payload = {
        "manga_id": args.manga_id,
        "requested_chapters": len(chapters),
        "all_chapters": bool(args.all_chapters),
        **_summarize_ingest_results(results),
        "chapters": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare_human_gold(args: argparse.Namespace) -> None:
    selected_assets = _select_catalog_assets(args)
    thresholds = _build_thresholds(args)
    predictions = _load_predictions_with_optional_catalog(args.predictions, args.db) if args.predictions else []
    selections = select_human_gold_pages(
        selected_assets,
        predictions=predictions,
        page_limit=args.page_limit,
        thresholds=thresholds,
    )
    tasks = build_human_gold_tasks(
        selections,
        predictions=predictions,
        image_base_url=args.image_base_url,
        local_files_document_root=args.local_files_document_root,
    )
    write_label_studio_tasks(tasks, args.tasks_out)
    write_human_gold_manifest(selections, args.manifest_out)
    payload = {
        **summarize_human_gold_batch(selections),
        "tasks_output": args.tasks_out,
        "manifest_output": args.manifest_out,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_teacher(args: argparse.Namespace) -> None:
    predictor = load_teacher_predictor(args.predictor, args.predictor_config)
    selected_assets = _select_catalog_assets(args)
    predictions, failures = run_teacher_predictions(
        predictor,
        selected_assets,
        continue_on_error=args.continue_on_error,
    )
    write_teacher_predictions(predictions, args.out)
    payload = {
        "predictor": args.predictor,
        "selected_assets": len(selected_assets),
        "predictions": len(predictions),
        "failed_assets": len(failures),
        "output": args.out,
        "failures": [failure.to_dict() for failure in failures],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_teacher_cycle_command(args: argparse.Namespace) -> None:
    predictor = load_teacher_predictor(args.predictor, args.predictor_config)
    selected_assets = _select_catalog_assets(args)
    thresholds = _build_thresholds(args)
    exception_policy = _build_exception_review_policy(args)
    artifacts = run_teacher_cycle(
        predictor,
        selected_assets,
        predictions_output_path=args.predictions_out,
        silver_output_path=args.silver_out,
        review_output_path=args.review_out,
        thresholds=thresholds,
        continue_on_error=args.continue_on_error,
        page_limit=args.page_limit,
        image_base_url=args.image_base_url,
        local_files_document_root=args.local_files_document_root,
        rejected_output_path=args.rejected_out,
        review_decisions_output_path=args.review_decisions_out,
        review_strategy=args.review_strategy,
        exception_policy=exception_policy,
    )
    payload = {
        "predictor": args.predictor,
        "selected_assets": len(selected_assets),
        **artifacts.summary(),
        "predictions_output": args.predictions_out,
        "silver_output": args.silver_out,
        "review_output": args.review_out,
        "rejected_output": args.rejected_out,
        "review_decisions_output": args.review_decisions_out,
        "failures": [failure.to_dict() for failure in artifacts.failures],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_predictions_command(args: argparse.Namespace) -> None:
    gold_records = load_gold_manifest_records(_resolve_input_path(args.gold, must_exist=True))
    predictions = _load_predictions_with_optional_catalog(args.predictions, args.db)
    summary = evaluate_predictions(gold_records, predictions, iou_threshold=args.iou_threshold)
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["output"] = args.out
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _ingest_descriptors(
    adapter: ChapterApiAdapter,
    descriptors: Sequence[ChapterDescriptor],
    db_path: str,
    storage_root: str,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    pipeline = IngestionPipeline(adapter=adapter, catalog=CatalogDatabase(_resolve_input_path(db_path)), storage_root=storage_root)
    ingested: list[dict[str, Any]] = []
    for descriptor in descriptors:
        report = pipeline.ingest_chapter_with_report(
            descriptor.chapter_id,
            continue_on_error=continue_on_error,
        )
        status = "ok"
        if report.errors and report.assets:
            status = "partial"
        elif report.errors:
            status = "failed"
        ingested.append(
            {
                "series_id": descriptor.series_id,
                "chapter_id": descriptor.chapter_id,
                "status": status,
                "pages": len(report.assets),
                "page_errors": [error.to_dict() for error in report.errors],
                "first_asset_id": report.assets[0].asset_id if report.assets else None,
                "title": descriptor.metadata.get("manga_title") or descriptor.metadata.get("title"),
                "chapter_title": descriptor.metadata.get("chapter_title") or descriptor.metadata.get("last_chapter"),
            }
        )
    return ingested


def _summarize_ingest_results(results: Sequence[dict[str, Any]]) -> dict[str, int]:
    succeeded = sum(1 for item in results if item.get("status") == "ok")
    partial = sum(1 for item in results if item.get("status") == "partial")
    failed = sum(1 for item in results if item.get("status") == "failed")
    pages = sum(int(item.get("pages") or 0) for item in results)
    page_errors = sum(len(item.get("page_errors") or []) for item in results)
    return {
        "succeeded_chapters": succeeded,
        "partial_chapters": partial,
        "failed_chapters": failed,
        "ingested_pages": pages,
        "page_errors": page_errors,
    }


def _build_consumet_adapter(args: argparse.Namespace) -> ConsumetMangahereAdapter:
    return ConsumetMangahereAdapter(
        base_url=args.base_url,
        referer=args.referer,
        timeout_s=args.timeout_s,
        metadata_timeout_s=args.metadata_timeout_s,
        image_timeout_s=args.image_timeout_s,
        source_id=args.source_id,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
    )


def _select_catalog_assets(args: argparse.Namespace) -> list[PageAsset]:
    catalog = CatalogDatabase(_resolve_input_path(args.db))
    return select_page_assets(
        catalog.list_page_assets(),
        source_id=args.source_id,
        series_id=args.series_id,
        chapter_id=args.chapter_id,
        limit=args.limit,
        include_duplicates=args.include_duplicates,
    )


def _export_assets_manifest(db_path: str, output_path: str) -> None:
    catalog = CatalogDatabase(_resolve_input_path(db_path))
    write_page_assets_manifest(catalog.list_page_assets(), output_path)
    print(f"Wrote page assets manifest to {output_path}")


def _build_label_studio_tasks(
    db_path: str,
    output_path: str,
    image_base_url: str | None,
    local_files_document_root: str | None,
) -> None:
    catalog = CatalogDatabase(_resolve_input_path(db_path))
    tasks = build_label_studio_tasks(
        catalog.list_page_assets(),
        image_base_url=image_base_url,
        local_files_document_root=local_files_document_root,
    )
    write_label_studio_tasks(tasks, output_path)
    print(f"Wrote {len(tasks)} Label Studio tasks to {output_path}")


def _convert_label_studio_export(input_path: str, output_path: str) -> None:
    payload = json.loads(_resolve_input_path(input_path, must_exist=True).read_text(encoding="utf-8"))
    records = convert_label_studio_export(payload)
    write_dataset_manifest(records, output_path)
    print(f"Wrote {len(records)} dataset manifest records to {output_path}")


def _build_silver_manifest(args: argparse.Namespace) -> None:
    thresholds = _build_thresholds(args)
    predictions = _load_predictions_with_optional_catalog(args.input, args.db)
    records, rejected = build_silver_manifest(predictions, thresholds=thresholds)
    write_dataset_manifest(records, args.out)
    if args.rejected_out:
        write_filter_decisions(rejected, args.rejected_out)
    summary = summarize_predictions(predictions, thresholds=thresholds)
    summary["silver_manifest_path"] = args.out
    if args.rejected_out:
        summary["rejected_path"] = args.rejected_out
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_review_queue(args: argparse.Namespace) -> None:
    thresholds = _build_thresholds(args)
    records = read_prediction_records(_resolve_input_path(args.predictions, must_exist=True))
    catalog = CatalogDatabase(_resolve_input_path(args.db))
    asset_ids = [str(record.get("asset_id") or "").strip() for record in records]
    page_assets = catalog.list_page_assets_by_ids(asset_ids)
    page_assets_by_id = {asset.asset_id: asset for asset in page_assets}
    predictions = teacher_predictions_from_records(records, page_assets_by_id=page_assets_by_id)

    if args.review_strategy == 'exception-audit':
        policy = _build_exception_review_policy(args)
        tasks, decisions = build_review_by_exception_tasks(
            page_assets,
            predictions,
            thresholds=thresholds,
            policy=policy,
            image_base_url=args.image_base_url,
            local_files_document_root=args.local_files_document_root,
        )
        summary = summarize_exception_review(decisions)
        if args.decisions_out:
            write_exception_review_decisions(decisions, args.decisions_out)
    else:
        tasks, decisions = build_review_tasks(
            page_assets,
            predictions,
            page_limit=args.page_limit,
            thresholds=thresholds,
            image_base_url=args.image_base_url,
            local_files_document_root=args.local_files_document_root,
        )
        summary = {
            'review_tasks': len(tasks),
            'review_decisions': len(decisions),
        }
        if args.decisions_out:
            write_filter_decisions(decisions, args.decisions_out)

    write_label_studio_tasks(tasks, args.out)
    summary.update(
        {
            'review_strategy': args.review_strategy,
            'review_tasks': len(tasks),
            'review_decisions': len(decisions),
            'output': args.out,
            'decisions_output': args.decisions_out,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _plan_training(config_path: str) -> None:
    pipeline = OcrTrainingPipeline.from_path(_resolve_input_path(config_path, must_exist=True))
    print(json.dumps(pipeline.summary(), ensure_ascii=False, indent=2))


def _load_adapter(import_path: str, config_path: str | None) -> ChapterApiAdapter:
    module_name, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)

    config: dict[str, Any] = {}
    if config_path:
        config = json.loads(_resolve_input_path(config_path, must_exist=True).read_text(encoding="utf-8"))

    adapter = adapter_class(**config)
    if not isinstance(adapter, ChapterApiAdapter):
        raise TypeError(f"{import_path} is not a ChapterApiAdapter")
    return adapter


def _load_predictions_with_optional_catalog(
    input_path: str | None,
    db_path: str | None,
):
    if not input_path:
        return []
    records = read_prediction_records(_resolve_input_path(input_path, must_exist=True))
    page_assets_by_id: dict[str, Any] = {}
    if db_path:
        asset_ids = [str(record.get("asset_id") or "").strip() for record in records]
        catalog = CatalogDatabase(_resolve_input_path(db_path))
        page_assets = catalog.list_page_assets_by_ids(asset_ids)
        page_assets_by_id = {asset.asset_id: asset for asset in page_assets}
    return teacher_predictions_from_records(records, page_assets_by_id=page_assets_by_id)


if __name__ == "__main__":
    main()
