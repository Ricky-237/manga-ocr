from __future__ import annotations

import json
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib import error, parse, request

from ..domain import Domain


@dataclass(slots=True)
class ChapterDescriptor:
    chapter_id: str
    series_id: str
    source_id: str = "chapter-api"
    domain: Domain = Domain.MANGA
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChapterListing:
    chapters: list[ChapterDescriptor]
    next_cursor: str | None = None


@dataclass(slots=True)
class PageRef:
    source_id: str
    series_id: str
    chapter_id: str
    page_index: int
    image_ref: str
    domain: Domain = Domain.MANGA
    metadata: dict[str, Any] = field(default_factory=dict)


class ChapterApiAdapter(ABC):
    @abstractmethod
    def listChapters(self, cursor: str | None = None) -> ChapterListing:
        raise NotImplementedError

    @abstractmethod
    def getChapterPages(self, chapter_id: str) -> Sequence[PageRef]:
        raise NotImplementedError

    @abstractmethod
    def downloadPage(self, page_ref: PageRef) -> bytes:
        raise NotImplementedError

    def list_chapters(self, cursor: str | None = None) -> ChapterListing:
        return self.listChapters(cursor=cursor)

    def get_chapter_pages(self, chapter_id: str) -> Sequence[PageRef]:
        return self.getChapterPages(chapter_id=chapter_id)

    def download_page(self, page_ref: PageRef) -> bytes:
        return self.downloadPage(page_ref=page_ref)


class SimpleHttpResponse:
    def __init__(self, content: bytes, status_code: int, url: str) -> None:
        self.content = content
        self.status_code = status_code
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            preview = self.content.decode("utf-8", errors="ignore")[:200]
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}: {preview}")

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class SimpleHttpSession:
    def __init__(
        self,
        default_headers: dict[str, str] | None = None,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        retry_on_status_codes: Sequence[int] | None = None,
    ) -> None:
        self.default_headers = default_headers or {}
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_s = max(retry_backoff_s, 0.0)
        self.retry_on_status_codes = tuple(retry_on_status_codes or (408, 429, 500, 502, 503, 504))

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_backoff_s: float | None = None,
        retry_on_status_codes: Sequence[int] | None = None,
    ) -> SimpleHttpResponse:
        final_url = self._with_query(url, params)
        merged_headers = dict(self.default_headers)
        if headers:
            merged_headers.update(headers)

        max_retries = self.max_retries if retries is None else max(retries, 0)
        backoff_s = self.retry_backoff_s if retry_backoff_s is None else max(retry_backoff_s, 0.0)
        retry_status_codes = tuple(retry_on_status_codes or self.retry_on_status_codes)
        total_attempts = 1 + max_retries

        for attempt_index in range(total_attempts):
            req = request.Request(final_url, headers=merged_headers, method="GET")
            try:
                with request.urlopen(req, timeout=timeout) as response:
                    return SimpleHttpResponse(response.read(), int(response.status), final_url)
            except error.HTTPError as exc:
                response = SimpleHttpResponse(exc.read(), int(exc.code), final_url)
                if response.status_code in retry_status_codes and attempt_index + 1 < total_attempts:
                    self._sleep_before_retry(attempt_index, backoff_s)
                    continue
                return response
            except error.URLError as exc:
                if attempt_index + 1 < total_attempts:
                    self._sleep_before_retry(attempt_index, backoff_s)
                    continue
                if self._is_timeout_error(exc):
                    raise RuntimeError(self._format_timeout_message(final_url, timeout, total_attempts)) from exc
                raise RuntimeError(f"Failed to GET {final_url}: {exc.reason}") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt_index + 1 < total_attempts:
                    self._sleep_before_retry(attempt_index, backoff_s)
                    continue
                raise RuntimeError(self._format_timeout_message(final_url, timeout, total_attempts)) from exc
            except OSError as exc:
                if attempt_index + 1 < total_attempts:
                    self._sleep_before_retry(attempt_index, backoff_s)
                    continue
                if self._is_timeout_error(exc):
                    raise RuntimeError(self._format_timeout_message(final_url, timeout, total_attempts)) from exc
                raise RuntimeError(f"Failed to GET {final_url}: {exc}") from exc

        raise RuntimeError(f"Failed to GET {final_url}: exhausted retries")

    @staticmethod
    def _with_query(url: str, params: dict[str, Any] | None) -> str:
        if not params:
            return url
        query = parse.urlencode({key: value for key, value in params.items() if value is not None})
        separator = "&" if parse.urlparse(url).query else "?"
        return f"{url}{separator}{query}"

    @staticmethod
    def _sleep_before_retry(attempt_index: int, backoff_s: float) -> None:
        if backoff_s <= 0:
            return
        time.sleep(backoff_s * (2**attempt_index))

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, error.URLError):
            reason = exc.reason
            return isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
        return "timed out" in str(exc).lower()

    @staticmethod
    def _format_timeout_message(url: str, timeout: float | None, attempts: int) -> str:
        timeout_part = f"timeout={timeout}s" if timeout is not None else "timeout=default"
        return f"Timed out while GET {url} after {attempts} attempt(s) ({timeout_part})"


class HttpChapterApiAdapter(ChapterApiAdapter, ABC):
    def __init__(
        self,
        headers: dict[str, str] | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.headers = headers or {}
        self.timeout_s = timeout_s
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_s = max(retry_backoff_s, 0.0)
        self._session = SimpleHttpSession(
            default_headers=self.headers,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )

    def listChapters(self, cursor: str | None = None) -> ChapterListing:
        url, params = self._list_chapters_request(cursor)
        payload = self._get_json(url, params=params)
        return self._parse_list_chapters_response(payload)

    def getChapterPages(self, chapter_id: str) -> Sequence[PageRef]:
        url, params = self._chapter_pages_request(chapter_id)
        payload = self._get_json(url, params=params)
        return self._parse_chapter_pages_response(chapter_id, payload)

    def downloadPage(self, page_ref: PageRef) -> bytes:
        url, params = self._page_download_request(page_ref)
        response = self._session.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        return response.content

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self._session.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    @abstractmethod
    def _list_chapters_request(self, cursor: str | None) -> tuple[str, dict[str, Any] | None]:
        raise NotImplementedError

    @abstractmethod
    def _parse_list_chapters_response(self, payload: Any) -> ChapterListing:
        raise NotImplementedError

    @abstractmethod
    def _chapter_pages_request(self, chapter_id: str) -> tuple[str, dict[str, Any] | None]:
        raise NotImplementedError

    @abstractmethod
    def _parse_chapter_pages_response(self, chapter_id: str, payload: Any) -> Sequence[PageRef]:
        raise NotImplementedError

    @abstractmethod
    def _page_download_request(self, page_ref: PageRef) -> tuple[str, dict[str, Any] | None]:
        raise NotImplementedError
