from __future__ import annotations

import unittest

from manga_ocr.domain import DatasetSplit
from manga_ocr.splits import SeriesSplitStrategy


class SplitTests(unittest.TestCase):
    def test_series_assignment_is_deterministic(self) -> None:
        strategy = SeriesSplitStrategy(salt="stable")
        first = strategy.assign("series-abc")
        second = strategy.assign("series-abc")
        self.assertEqual(first, second)

    def test_all_splits_are_possible(self) -> None:
        strategy = SeriesSplitStrategy(train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, salt="coverage")
        assigned = {strategy.assign(f"series-{index}") for index in range(200)}
        self.assertEqual(assigned, {DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST})


if __name__ == "__main__":
    unittest.main()
