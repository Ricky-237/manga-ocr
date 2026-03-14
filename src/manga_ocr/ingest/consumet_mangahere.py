from __future__ import annotations

from typing import Any, Sequence

from ..domain import Domain
from .adapters import ChapterDescriptor, ChapterListing, HttpChapterApiAdapter, PageRef

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
_DEFAULT_LANGUAGE = "en-US,en;q=0.9"
_DEFAULT_JSON_ACCEPT = "application/json,text/plain,*/*"
_DEFAULT_IMAGE_ACCEPT = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"


class ConsumetMangahereAdapter(HttpChapterApiAdapter):
    def __init__(
        self,
        base_url: str = "https://api-consumet-org-extension.vercel.app",
        referer: str = "https://mangahere.com/",
        timeout_s: float = 30.0,
        metadata_timeout_s: float | None = None,
        image_timeout_s: float | None = None,
        source_id: str = "consumet-mangahere",
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.origin = referer.rstrip("/")
        self.source_id = source_id
        self.metadata_timeout_s = metadata_timeout_s if metadata_timeout_s is not None else timeout_s
        self.image_timeout_s = image_timeout_s if image_timeout_s is not None else max(self.metadata_timeout_s, 90.0)
        super().__init__(
            headers=self._build_json_headers(),
            timeout_s=self.metadata_timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )

    def listChapters(self, cursor: str | None = None) -> ChapterListing:
        page = self._cursor_to_page(cursor)
        url, params = self._list_chapters_request(str(page))
        latest_payload = self._get_json(url, params=params)
        descriptors: list[ChapterDescriptor] = []
        for manga in latest_payload:
            manga_id = str(manga.get("id") or "").strip()
            if not manga_id:
                continue
            info = self.getMangaInfo(manga_id)
            chapters = self._chapter_descriptors_from_info(info, limit=1)
            if not chapters:
                continue
            chapter = chapters[0]
            chapter.metadata.update(
                {
                    "title": manga.get("title") or chapter.metadata.get("manga_title"),
                    "cover_image": manga.get("image") or chapter.metadata.get("cover_image"),
                    "status": manga.get("status") or chapter.metadata.get("status"),
                    "rating": manga.get("rating") or chapter.metadata.get("rating"),
                    "last_chapter": manga.get("last_chapter") or chapter.metadata.get("chapter_title"),
                    "last_update": manga.get("last_update"),
                }
            )
            descriptors.append(chapter)
        next_cursor = str(page + 1) if latest_payload else None
        return ChapterListing(chapters=descriptors, next_cursor=next_cursor)

    def list_latest_chapters(self, page: int = 1, limit: int | None = None) -> ChapterListing:
        listing = self.listChapters(str(page))
        if limit is None:
            return listing
        return ChapterListing(chapters=listing.chapters[:limit], next_cursor=listing.next_cursor)

    def getMangaInfo(self, manga_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/info"
        return self._get_json(url, params={"id": manga_id})

    def get_manga_info(self, manga_id: str) -> dict[str, Any]:
        return self.getMangaInfo(manga_id)

    def get_manga_chapters(self, manga_id: str, limit: int | None = None) -> list[ChapterDescriptor]:
        info = self.getMangaInfo(manga_id)
        return self._chapter_descriptors_from_info(info, limit=limit)

    def getChapterPages(self, chapter_id: str) -> Sequence[PageRef]:
        url, params = self._chapter_pages_request(chapter_id)
        payload = self._get_json(url, params=params)
        return self._parse_chapter_pages_response(chapter_id, payload)

    def downloadPage(self, page_ref: PageRef) -> bytes:
        url, params = self._page_download_request(page_ref)

        primary_headers = self._build_image_headers(use_page_headers=False, page_ref=page_ref)
        response = self._session.get(url, params=params, headers=primary_headers, timeout=self.image_timeout_s)
        if response.status_code < 400:
            return response.content

        fallback_referer = str((page_ref.metadata.get("image_headers") or {}).get("Referer") or "").strip()
        if response.status_code == 403 and fallback_referer and fallback_referer != self.referer:
            fallback_headers = self._build_image_headers(use_page_headers=True, page_ref=page_ref)
            fallback_response = self._session.get(
                url,
                params=params,
                headers=fallback_headers,
                timeout=self.image_timeout_s,
            )
            if fallback_response.status_code < 400:
                return fallback_response.content
            fallback_response.raise_for_status()

        response.raise_for_status()
        return response.content

    def _list_chapters_request(self, cursor: str | None) -> tuple[str, dict[str, Any] | None]:
        return f"{self.base_url}/latest", {"page": self._cursor_to_page(cursor)}

    def _parse_list_chapters_response(self, payload: Any) -> ChapterListing:
        chapters = [
            ChapterDescriptor(
                chapter_id=str(item.get("id") or ""),
                series_id=str(item.get("id") or ""),
                source_id=self.source_id,
                domain=Domain.MANGA,
                metadata={
                    "title": item.get("title"),
                    "cover_image": item.get("image"),
                    "status": item.get("status"),
                    "rating": item.get("rating"),
                    "last_chapter": item.get("last_chapter"),
                    "last_update": item.get("last_update"),
                },
            )
            for item in payload
            if item.get("id")
        ]
        return ChapterListing(chapters=chapters)

    def _chapter_pages_request(self, chapter_id: str) -> tuple[str, dict[str, Any] | None]:
        return f"{self.base_url}/read", {"chapterId": chapter_id}

    def _parse_chapter_pages_response(self, chapter_id: str, payload: Any) -> Sequence[PageRef]:
        series_id = str(chapter_id).split("/", 1)[0]
        pages: list[PageRef] = []
        for item in payload.get("pages", []):
            pages.append(
                PageRef(
                    source_id=self.source_id,
                    series_id=series_id,
                    chapter_id=chapter_id,
                    page_index=int(item.get("page", 0)),
                    image_ref=str(item.get("img") or ""),
                    domain=Domain.MANGA,
                    metadata={
                        "page_number": item.get("page"),
                        "image_headers": item.get("headerForImage") or {},
                        "next_chapter": payload.get("next"),
                        "prev_chapter": payload.get("prev"),
                    },
                )
            )
        return sorted(pages, key=lambda page: page.page_index)

    def _page_download_request(self, page_ref: PageRef) -> tuple[str, dict[str, Any] | None]:
        return page_ref.image_ref, None

    def _chapter_descriptors_from_info(
        self,
        info_payload: dict[str, Any],
        limit: int | None = None,
    ) -> list[ChapterDescriptor]:
        manga_id = str(info_payload.get("id") or "").strip()
        if not manga_id:
            return []
        chapters = info_payload.get("chapters") or []
        if limit is not None:
            chapters = chapters[: max(limit, 0)]
        descriptors: list[ChapterDescriptor] = []
        for chapter in chapters:
            chapter_id = str(chapter.get("id") or "").strip()
            if not chapter_id:
                continue
            descriptors.append(
                ChapterDescriptor(
                    chapter_id=chapter_id,
                    series_id=manga_id,
                    source_id=self.source_id,
                    domain=Domain.MANGA,
                    metadata={
                        "manga_title": info_payload.get("title"),
                        "cover_image": info_payload.get("image"),
                        "description": info_payload.get("description"),
                        "status": info_payload.get("status"),
                        "rating": info_payload.get("rating"),
                        "genres": info_payload.get("genres") or [],
                        "authors": info_payload.get("authors") or [],
                        "chapter_title": chapter.get("title"),
                        "is_new": bool(chapter.get("isNew", False)),
                    },
                )
            )
        return descriptors

    def _build_json_headers(self) -> dict[str, str]:
        return {
            "Referer": self.referer,
            "Origin": self.origin,
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": _DEFAULT_JSON_ACCEPT,
            "Accept-Language": _DEFAULT_LANGUAGE,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _build_image_headers(self, use_page_headers: bool, page_ref: PageRef) -> dict[str, str]:
        headers = {
            "Referer": self.referer,
            "Origin": self.origin,
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": _DEFAULT_IMAGE_ACCEPT,
            "Accept-Language": _DEFAULT_LANGUAGE,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if use_page_headers:
            for key, value in (page_ref.metadata.get("image_headers") or {}).items():
                headers[str(key)] = str(value)
        return headers

    @staticmethod
    def _cursor_to_page(cursor: str | None) -> int:
        if cursor is None:
            return 1
        try:
            page = int(cursor)
        except ValueError as exc:
            raise ValueError(f"Invalid latest-page cursor: {cursor!r}") from exc
        return max(page, 1)
