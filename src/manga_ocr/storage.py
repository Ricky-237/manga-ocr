from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Sequence

from .domain import Domain, PageAsset

PAGE_ASSET_COLUMNS = """
    source_id,
    series_id,
    chapter_id,
    page_index,
    image_path,
    sha256,
    phash,
    width,
    height,
    fetched_at,
    domain,
    metadata_json,
    canonical_asset_id,
    is_duplicate
"""


class CatalogDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS page_assets (
                    asset_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    phash TEXT,
                    width INTEGER,
                    height INTEGER,
                    fetched_at TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    canonical_asset_id TEXT,
                    is_duplicate INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(source_id, series_id, chapter_id, page_index)
                );

                CREATE INDEX IF NOT EXISTS idx_page_assets_series ON page_assets(series_id);
                CREATE INDEX IF NOT EXISTS idx_page_assets_sha256 ON page_assets(sha256);
                """
            )
            connection.commit()

    def upsert_page_asset(self, asset: PageAsset) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO page_assets (
                    asset_id,
                    source_id,
                    series_id,
                    chapter_id,
                    page_index,
                    image_path,
                    sha256,
                    phash,
                    width,
                    height,
                    fetched_at,
                    domain,
                    metadata_json,
                    canonical_asset_id,
                    is_duplicate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    image_path = excluded.image_path,
                    sha256 = excluded.sha256,
                    phash = excluded.phash,
                    width = excluded.width,
                    height = excluded.height,
                    fetched_at = excluded.fetched_at,
                    domain = excluded.domain,
                    metadata_json = excluded.metadata_json,
                    canonical_asset_id = excluded.canonical_asset_id,
                    is_duplicate = excluded.is_duplicate
                """,
                (
                    asset.asset_id,
                    asset.source_id,
                    asset.series_id,
                    asset.chapter_id,
                    asset.page_index,
                    asset.image_path,
                    asset.sha256,
                    asset.phash,
                    asset.width,
                    asset.height,
                    asset.fetched_at,
                    asset.domain.value,
                    json.dumps(asset.metadata, ensure_ascii=False),
                    asset.canonical_asset_id,
                    1 if asset.is_duplicate else 0,
                ),
            )
            connection.commit()

    def find_asset_by_sha256(self, sha256: str) -> PageAsset | None:
        query = f"""
            SELECT {PAGE_ASSET_COLUMNS}
            FROM page_assets
            WHERE sha256 = ?
            ORDER BY is_duplicate ASC, asset_id ASC
            LIMIT 1
        """
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(query, (sha256,)).fetchone()
        return self._row_to_asset(row) if row else None

    def find_asset(self, asset_id: str) -> PageAsset | None:
        query = f"""
            SELECT {PAGE_ASSET_COLUMNS}
            FROM page_assets
            WHERE asset_id = ?
        """
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(query, (asset_id,)).fetchone()
        return self._row_to_asset(row) if row else None

    def list_page_assets(self) -> list[PageAsset]:
        query = f"""
            SELECT {PAGE_ASSET_COLUMNS}
            FROM page_assets
            ORDER BY source_id, series_id, chapter_id, page_index
        """
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(query).fetchall()
        return [self._row_to_asset(row) for row in rows]

    def list_page_assets_by_ids(self, asset_ids: Sequence[str]) -> list[PageAsset]:
        normalized = [str(asset_id) for asset_id in asset_ids if str(asset_id).strip()]
        if not normalized:
            return []
        placeholders = ", ".join("?" for _ in normalized)
        query = f"""
            SELECT {PAGE_ASSET_COLUMNS}
            FROM page_assets
            WHERE asset_id IN ({placeholders})
        """
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(query, normalized).fetchall()
        assets = [self._row_to_asset(row) for row in rows]
        assets.sort(key=lambda asset: normalized.index(asset.asset_id))
        return assets

    @staticmethod
    def _row_to_asset(row: Iterable[object]) -> PageAsset:
        (
            source_id,
            series_id,
            chapter_id,
            page_index,
            image_path,
            sha256,
            phash,
            width,
            height,
            fetched_at,
            domain,
            metadata_json,
            canonical_asset_id,
            is_duplicate,
        ) = row
        return PageAsset(
            source_id=str(source_id),
            series_id=str(series_id),
            chapter_id=str(chapter_id),
            page_index=int(page_index),
            image_path=str(image_path),
            sha256=str(sha256),
            phash=str(phash) if phash is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            fetched_at=str(fetched_at),
            domain=Domain(str(domain)),
            metadata=json.loads(str(metadata_json) if metadata_json else "{}"),
            canonical_asset_id=str(canonical_asset_id) if canonical_asset_id else None,
            is_duplicate=bool(is_duplicate),
        )
