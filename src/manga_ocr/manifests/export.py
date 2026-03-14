from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..domain import DatasetManifestRecord, PageAsset


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str | Path) -> list[dict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_page_assets_manifest(page_assets: Iterable[PageAsset], output_path: str | Path) -> None:
    write_jsonl((asset.to_dict() for asset in page_assets), output_path)


def write_dataset_manifest(records: Iterable[DatasetManifestRecord], output_path: str | Path) -> None:
    write_jsonl((record.to_dict() for record in records), output_path)
