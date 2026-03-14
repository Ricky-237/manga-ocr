from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..domain import PageAsset
from ..image_utils import fingerprint_image, store_raw_image
from ..storage import CatalogDatabase
from .adapters import ChapterApiAdapter, PageRef


@dataclass(slots=True)
class IngestionResult:
    asset: PageAsset
    was_existing: bool


@dataclass(slots=True)
class IngestionError:
    stage: str
    chapter_id: str
    error: str
    page_index: int | None = None
    image_ref: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "stage": self.stage,
            "chapter_id": self.chapter_id,
            "page_index": self.page_index,
            "image_ref": self.image_ref,
            "error": self.error,
        }


@dataclass(slots=True)
class ChapterIngestionReport:
    chapter_id: str
    assets: list[PageAsset]
    errors: list[IngestionError]


class IngestionPipeline:
    def __init__(
        self,
        adapter: ChapterApiAdapter,
        catalog: CatalogDatabase,
        storage_root: str | Path,
    ) -> None:
        self.adapter = adapter
        self.catalog = catalog
        self.storage_root = Path(storage_root)

    def ingest_chapter(self, chapter_id: str) -> list[PageAsset]:
        return self.ingest_chapter_with_report(chapter_id=chapter_id, continue_on_error=False).assets

    def ingest_chapter_with_report(
        self,
        chapter_id: str,
        continue_on_error: bool = False,
    ) -> ChapterIngestionReport:
        self.catalog.initialize()
        try:
            page_refs = sorted(self.adapter.getChapterPages(chapter_id), key=lambda ref: ref.page_index)
        except Exception as exc:
            if not continue_on_error:
                raise
            return ChapterIngestionReport(
                chapter_id=chapter_id,
                assets=[],
                errors=[IngestionError(stage="list_pages", chapter_id=chapter_id, error=str(exc))],
            )

        assets: list[PageAsset] = []
        errors: list[IngestionError] = []
        for page_ref in page_refs:
            try:
                assets.append(self.ingest_page(page_ref).asset)
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append(
                    IngestionError(
                        stage="download_page",
                        chapter_id=chapter_id,
                        page_index=page_ref.page_index,
                        image_ref=page_ref.image_ref,
                        error=str(exc),
                    )
                )
        return ChapterIngestionReport(chapter_id=chapter_id, assets=assets, errors=errors)

    def ingest_page(self, page_ref: PageRef) -> IngestionResult:
        now = datetime.now(timezone.utc).isoformat()
        candidate_asset_id = (
            f"{page_ref.source_id}:{page_ref.series_id}:{page_ref.chapter_id}:{page_ref.page_index:04d}"
        )
        existing_asset = self.catalog.find_asset(candidate_asset_id)
        data = self.adapter.downloadPage(page_ref)
        fingerprint = fingerprint_image(data)

        canonical_asset = self.catalog.find_asset_by_sha256(fingerprint.sha256)
        is_duplicate = canonical_asset is not None and canonical_asset.asset_id != candidate_asset_id

        if canonical_asset is None:
            image_path = store_raw_image(
                self.storage_root,
                fingerprint.sha256,
                fingerprint.extension,
                data,
            )
            canonical_asset_id = None
            phash = fingerprint.phash
            width = fingerprint.width
            height = fingerprint.height
        else:
            image_path = Path(canonical_asset.image_path)
            canonical_asset_id = canonical_asset.asset_id
            phash = canonical_asset.phash or fingerprint.phash
            width = canonical_asset.width or fingerprint.width
            height = canonical_asset.height or fingerprint.height

        asset = PageAsset(
            source_id=page_ref.source_id,
            series_id=page_ref.series_id,
            chapter_id=page_ref.chapter_id,
            page_index=page_ref.page_index,
            image_path=str(image_path),
            sha256=fingerprint.sha256,
            phash=phash,
            width=width,
            height=height,
            fetched_at=now,
            domain=page_ref.domain,
            metadata=dict(page_ref.metadata),
            canonical_asset_id=canonical_asset_id,
            is_duplicate=is_duplicate,
        )
        self.catalog.upsert_page_asset(asset)
        return IngestionResult(asset=asset, was_existing=existing_asset is not None)
