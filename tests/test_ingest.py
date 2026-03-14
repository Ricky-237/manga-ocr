from __future__ import annotations

import base64
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from manga_ocr.domain import Domain
from manga_ocr.ingest import ChapterApiAdapter, ChapterListing, ConsumetMangahereAdapter, IngestionPipeline, PageRef
from manga_ocr.ingest.adapters import SimpleHttpSession
from manga_ocr.storage import CatalogDatabase


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZkvoAAAAASUVORK5CYII="
)


class FakeAdapter(ChapterApiAdapter):
    def __init__(self) -> None:
        self._pages = {
            "chapter-a": [
                PageRef(
                    source_id="api",
                    series_id="series-1",
                    chapter_id="chapter-a",
                    page_index=0,
                    image_ref="page-0",
                    domain=Domain.MANGA,
                ),
                PageRef(
                    source_id="api",
                    series_id="series-1",
                    chapter_id="chapter-a",
                    page_index=1,
                    image_ref="page-1",
                    domain=Domain.MANGA,
                ),
            ]
        }

    def listChapters(self, cursor: str | None = None) -> ChapterListing:
        raise NotImplementedError

    def getChapterPages(self, chapter_id: str):
        return list(self._pages[chapter_id])

    def downloadPage(self, page_ref: PageRef) -> bytes:
        return PNG_1X1


class FlakyAdapter(FakeAdapter):
    def downloadPage(self, page_ref: PageRef) -> bytes:
        if page_ref.page_index == 1:
            raise TimeoutError("The read operation timed out")
        return PNG_1X1


class DummyResponse:
    def __init__(self, content: bytes = b"image-bytes", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DummySession:
    def __init__(self, responses=None) -> None:
        self.calls: list[dict] = []
        self.responses = list(responses or [DummyResponse()])

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if self.responses:
            next_item = self.responses.pop(0)
            if isinstance(next_item, BaseException):
                raise next_item
            return next_item
        return DummyResponse()


class UrlOpenResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self._content = content
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._content


class FakeConsumetAdapter(ConsumetMangahereAdapter):
    def __init__(self) -> None:
        self.base_url = "https://api-consumet-org-extension.vercel.app"
        self.referer = "https://mangahere.com/"
        self.origin = "https://mangahere.com"
        self.source_id = "consumet-mangahere"
        self.metadata_timeout_s = 30.0
        self.image_timeout_s = 90.0
        self.timeout_s = self.metadata_timeout_s
        self.max_retries = 2
        self.retry_backoff_s = 1.0
        self.headers = self._build_json_headers()
        self._session = DummySession()
        self.payloads = {
            (f"{self.base_url}/latest", (("page", 1),)): [
                {
                    "dat_id": 15539,
                    "id": "seishun_otome_banchou",
                    "title": "Seishun Otome Banchou!",
                    "image": "https://fmcdn.mangahere.com/store/manga/15539/cover.jpg",
                    "status": "Completed",
                    "rating": 4.78,
                    "last_chapter": "Ch.310",
                    "last_update": "13 minute ago",
                }
            ],
            (f"{self.base_url}/info", (("id", "seishun_otome_banchou"),)): {
                "id": "seishun_otome_banchou",
                "title": "Seishun Otome Banchou!",
                "image": "https://fmcdn.mangahere.com/store/manga/15539/cover.jpg",
                "description": "desc",
                "status": "vol",
                "rating": 4.84,
                "genres": ["Action"],
                "authors": ["ODA Eiichiro"],
                "chapters": [
                    {
                        "id": "seishun_otome_banchou/v1/c310",
                        "title": "Vol.1 Ch.310",
                        "isNew": False,
                    },
                    {
                        "id": "seishun_otome_banchou/v1/c309",
                        "title": "Vol.1 Ch.309",
                        "isNew": False,
                    },
                ],
            },
            (f"{self.base_url}/read", (("chapterId", "seishun_otome_banchou/v1/c310"),)): {
                "next": {"id": "seishun_otome_banchou/v1/c311", "title": "Vol.1 Ch.311"},
                "prev": {"id": "seishun_otome_banchou/v1/c309", "title": "Vol.1 Ch.309"},
                "pages": [
                    {
                        "page": 0,
                        "img": "https://zjcdn.mangahere.org/store/manga/15539/1-310/compressed/fp_001.jpg",
                        "headerForImage": {
                            "Referer": "https://newm.mangahere.cc/manga/seishun_otome_banchou/v1/c310/"
                        },
                    }
                ],
            },
        }

    def _get_json(self, url, params=None):
        normalized = tuple(sorted((params or {}).items()))
        return self.payloads[(url, normalized)]


class IngestTests(unittest.TestCase):
    def test_ingest_tracks_duplicates_by_sha256(self) -> None:
        scratch_root = Path(".tmp-tests")
        scratch_root.mkdir(exist_ok=True)
        temp_dir = scratch_root / f"ingest-{uuid4().hex}"
        temp_dir.mkdir(parents=True)
        try:
            catalog = CatalogDatabase(temp_dir / "catalog.sqlite")
            pipeline = IngestionPipeline(FakeAdapter(), catalog, temp_dir / "data")

            assets = pipeline.ingest_chapter("chapter-a")

            self.assertEqual(len(assets), 2)
            self.assertEqual(assets[0].sha256, assets[1].sha256)
            self.assertFalse(assets[0].is_duplicate)
            self.assertTrue(assets[1].is_duplicate)
            self.assertEqual(assets[1].canonical_asset_id, assets[0].asset_id)
            self.assertTrue(Path(assets[0].image_path).exists())
            self.assertEqual(assets[0].width, 1)
            self.assertEqual(assets[0].height, 1)

            stored = catalog.list_page_assets()
            self.assertEqual(len(stored), 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ingest_chapter_with_report_collects_page_errors_when_continuing(self) -> None:
        scratch_root = Path(".tmp-tests")
        scratch_root.mkdir(exist_ok=True)
        temp_dir = scratch_root / f"ingest-report-{uuid4().hex}"
        temp_dir.mkdir(parents=True)
        try:
            catalog = CatalogDatabase(temp_dir / "catalog.sqlite")
            pipeline = IngestionPipeline(FlakyAdapter(), catalog, temp_dir / "data")

            report = pipeline.ingest_chapter_with_report("chapter-a", continue_on_error=True)

            self.assertEqual(len(report.assets), 1)
            self.assertEqual(len(report.errors), 1)
            self.assertEqual(report.errors[0].stage, "download_page")
            self.assertEqual(report.errors[0].page_index, 1)
            self.assertIn("timed out", report.errors[0].error.lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_simple_http_session_retries_after_timeout(self) -> None:
        attempts: list[float | None] = []

        def fake_urlopen(req, timeout=None):
            attempts.append(timeout)
            if len(attempts) == 1:
                raise TimeoutError("The read operation timed out")
            return UrlOpenResponse(b"ok")

        session = SimpleHttpSession(max_retries=1, retry_backoff_s=0.0)
        with patch("manga_ocr.ingest.adapters.request.urlopen", side_effect=fake_urlopen):
            response = session.get("https://example.com/image.jpg", timeout=12.0)

        self.assertEqual(response.content, b"ok")
        self.assertEqual(attempts, [12.0, 12.0])

    def test_consumet_list_chapters_resolves_latest_chapter_from_info(self) -> None:
        adapter = FakeConsumetAdapter()

        listing = adapter.listChapters()

        self.assertEqual(len(listing.chapters), 1)
        chapter = listing.chapters[0]
        self.assertEqual(chapter.chapter_id, "seishun_otome_banchou/v1/c310")
        self.assertEqual(chapter.series_id, "seishun_otome_banchou")
        self.assertEqual(chapter.metadata["title"], "Seishun Otome Banchou!")
        self.assertEqual(listing.next_cursor, "2")

    def test_consumet_get_manga_chapters_respects_limit(self) -> None:
        adapter = FakeConsumetAdapter()

        chapters = adapter.get_manga_chapters("seishun_otome_banchou", limit=1)

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].chapter_id, "seishun_otome_banchou/v1/c310")
        self.assertEqual(chapters[0].metadata["manga_title"], "Seishun Otome Banchou!")
        self.assertEqual(chapters[0].metadata["chapter_title"], "Vol.1 Ch.310")

    def test_consumet_get_chapter_pages_preserves_image_headers(self) -> None:
        adapter = FakeConsumetAdapter()

        pages = adapter.getChapterPages("seishun_otome_banchou/v1/c310")

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_index, 0)
        self.assertEqual(
            pages[0].metadata["image_headers"]["Referer"],
            "https://newm.mangahere.cc/manga/seishun_otome_banchou/v1/c310/",
        )

    def test_consumet_download_prefers_global_referer_and_browser_headers(self) -> None:
        adapter = FakeConsumetAdapter()
        adapter.image_timeout_s = 123.0
        adapter._session = DummySession([DummyResponse(content=b"ok")])
        page = PageRef(
            source_id="consumet-mangahere",
            series_id="seishun_otome_banchou",
            chapter_id="seishun_otome_banchou/v1/c310",
            page_index=0,
            image_ref="https://zjcdn.mangahere.org/store/manga/15539/1-310/compressed/fp_001.jpg",
            domain=Domain.MANGA,
            metadata={
                "image_headers": {
                    "Referer": "https://newm.mangahere.cc/manga/seishun_otome_banchou/v1/c310/"
                }
            },
        )

        content = adapter.downloadPage(page)

        self.assertEqual(content, b"ok")
        self.assertEqual(adapter._session.calls[0]["headers"]["Referer"], "https://mangahere.com/")
        self.assertIn("Mozilla/5.0", adapter._session.calls[0]["headers"]["User-Agent"])
        self.assertIn("image/", adapter._session.calls[0]["headers"]["Accept"])
        self.assertEqual(adapter._session.calls[0]["timeout"], 123.0)

    def test_consumet_download_retries_with_page_referer_after_403(self) -> None:
        adapter = FakeConsumetAdapter()
        adapter._session = DummySession([
            DummyResponse(content=b"blocked", status_code=403),
            DummyResponse(content=b"ok", status_code=200),
        ])
        page = PageRef(
            source_id="consumet-mangahere",
            series_id="seishun_otome_banchou",
            chapter_id="seishun_otome_banchou/v1/c310",
            page_index=0,
            image_ref="https://zjcdn.mangahere.org/store/manga/15539/1-310/compressed/fp_001.jpg",
            domain=Domain.MANGA,
            metadata={
                "image_headers": {
                    "Referer": "https://newm.mangahere.cc/manga/seishun_otome_banchou/v1/c310/"
                }
            },
        )

        content = adapter.downloadPage(page)

        self.assertEqual(content, b"ok")
        self.assertEqual(len(adapter._session.calls), 2)
        self.assertEqual(adapter._session.calls[0]["headers"]["Referer"], "https://mangahere.com/")
        self.assertEqual(
            adapter._session.calls[1]["headers"]["Referer"],
            "https://newm.mangahere.cc/manga/seishun_otome_banchou/v1/c310/",
        )


if __name__ == "__main__":
    unittest.main()
