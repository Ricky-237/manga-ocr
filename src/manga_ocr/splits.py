from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .domain import DatasetSplit


@dataclass(slots=True)
class SeriesSplitStrategy:
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    salt: str = "manga-ocr-v1"

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if round(total, 6) != 1.0:
            raise ValueError("Split ratios must add up to 1.0")

    def assign(self, series_id: str) -> DatasetSplit:
        digest = hashlib.sha256(f"{self.salt}:{series_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        if bucket < self.train_ratio:
            return DatasetSplit.TRAIN
        if bucket < self.train_ratio + self.val_ratio:
            return DatasetSplit.VAL
        return DatasetSplit.TEST
