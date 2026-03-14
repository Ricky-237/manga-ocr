"""Core package for the manga OCR training scaffold."""

from .domain import DatasetManifestRecord, OcrBlock, OcrLineLabel, PageAsset, PolygonPoint

__all__ = [
    "DatasetManifestRecord",
    "OcrBlock",
    "OcrLineLabel",
    "PageAsset",
    "PolygonPoint",
]
