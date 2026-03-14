from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Domain(str, Enum):
    MANGA = "manga"
    WEBTOON = "webtoon"


class TextDirection(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class TextType(str, Enum):
    DIALOGUE = "dialogue"
    CAPTION = "caption"
    SFX = "sfx"
    UNKNOWN = "unknown"


class LabelSource(str, Enum):
    TEACHER = "teacher"
    HUMAN = "human"
    DISTILLED = "distilled"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class PolygonPoint:
    x: float
    y: float


@dataclass(slots=True)
class PageAsset:
    source_id: str
    series_id: str
    chapter_id: str
    page_index: int
    image_path: str
    sha256: str
    phash: str | None
    width: int | None
    height: int | None
    fetched_at: str
    domain: Domain = Domain.MANGA
    metadata: dict[str, Any] = field(default_factory=dict)
    canonical_asset_id: str | None = None
    is_duplicate: bool = False

    @property
    def asset_id(self) -> str:
        return f"{self.source_id}:{self.series_id}:{self.chapter_id}:{self.page_index:04d}"

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["asset_id"] = self.asset_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageAsset":
        return cls(
            source_id=data["source_id"],
            series_id=data["series_id"],
            chapter_id=data["chapter_id"],
            page_index=int(data["page_index"]),
            image_path=data["image_path"],
            sha256=data["sha256"],
            phash=data.get("phash"),
            width=data.get("width"),
            height=data.get("height"),
            fetched_at=data["fetched_at"],
            domain=Domain(data.get("domain", Domain.MANGA.value)),
            metadata=dict(data.get("metadata", {})),
            canonical_asset_id=data.get("canonical_asset_id"),
            is_duplicate=bool(data.get("is_duplicate", False)),
        )


@dataclass(slots=True)
class OcrLineLabel:
    polygon: list[PolygonPoint]
    transcript: str
    lang: str
    direction: TextDirection
    text_type: TextType
    confidence: float | None
    source: LabelSource

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class DatasetManifestRecord:
    image_path: str
    series_id: str
    domain: Domain
    tile_id: str | None
    polygon: list[PolygonPoint]
    transcript: str
    lang: str
    direction: TextDirection
    text_type: TextType
    split: DatasetSplit
    confidence: float | None = None
    source: LabelSource | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class OcrBlock:
    polygon: list[PolygonPoint]
    text: str
    lang: str
    direction: TextDirection
    confidence: float
    order_index: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)
