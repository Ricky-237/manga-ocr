from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from manga_ocr.domain import Domain, PageAsset
from manga_ocr.manifests import read_jsonl
from manga_ocr.storage import CatalogDatabase


class ExceptionReviewCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path('.tmp-tests')
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f'exception-cli-{uuid4().hex}'
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _insert_asset(self, db_path: Path, series_id: str) -> PageAsset:
        catalog = CatalogDatabase(db_path)
        catalog.initialize()
        asset = PageAsset(
            source_id='consumet-mangahere',
            series_id=series_id,
            chapter_id=f'{series_id}/c1',
            page_index=0,
            image_path=str(self.workdir / f'{series_id}.jpg'),
            sha256=f'sha-{series_id}',
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )
        catalog.upsert_page_asset(asset)
        return asset

    def test_build_review_queue_supports_exception_audit_strategy(self) -> None:
        from manga_ocr import cli

        db_path = self.workdir / 'catalog.sqlite'
        tasks_path = self.workdir / 'exception_review.json'
        decisions_path = self.workdir / 'exception_review_decisions.jsonl'
        predictions_path = self.workdir / 'teacher_predictions.jsonl'
        asset_a = self._insert_asset(db_path, 'series-a')
        asset_b = self._insert_asset(db_path, 'series-b')
        asset_c = self._insert_asset(db_path, 'series-c')
        predictions_path.write_text(
            '\n'.join(
                [
                    json.dumps(
                        {
                            'asset_id': asset_a.asset_id,
                            'series_id': asset_a.series_id,
                            'image_path': asset_a.image_path,
                            'domain': asset_a.domain.value,
                            'polygon': [{'x': 0, 'y': 0}, {'x': 20, 'y': 0}, {'x': 20, 'y': 20}],
                            'transcript': 'easy-a',
                            'lang': 'jp',
                            'direction': 'horizontal',
                            'text_type': 'dialogue',
                            'detection_confidence': 0.97,
                            'recognition_confidence': 0.96,
                            'script_confidence': 0.95,
                            'teacher_agreement': 0.94,
                            'min_text_height': 18,
                        }
                    ),
                    json.dumps(
                        {
                            'asset_id': asset_b.asset_id,
                            'series_id': asset_b.series_id,
                            'image_path': asset_b.image_path,
                            'domain': asset_b.domain.value,
                            'polygon': [{'x': 0, 'y': 0}, {'x': 20, 'y': 0}, {'x': 20, 'y': 20}],
                            'transcript': 'tiny',
                            'lang': 'jp',
                            'direction': 'horizontal',
                            'text_type': 'dialogue',
                            'detection_confidence': 0.40,
                            'recognition_confidence': 0.45,
                            'script_confidence': 0.42,
                            'teacher_agreement': 0.30,
                            'min_text_height': 6,
                        }
                    ),
                    json.dumps(
                        {
                            'asset_id': asset_c.asset_id,
                            'series_id': asset_c.series_id,
                            'image_path': asset_c.image_path,
                            'domain': asset_c.domain.value,
                            'polygon': [{'x': 0, 'y': 0}, {'x': 20, 'y': 0}, {'x': 20, 'y': 20}],
                            'transcript': 'easy-c',
                            'lang': 'jp',
                            'direction': 'horizontal',
                            'text_type': 'dialogue',
                            'detection_confidence': 0.97,
                            'recognition_confidence': 0.96,
                            'script_confidence': 0.95,
                            'teacher_agreement': 0.94,
                            'min_text_height': 18,
                        }
                    ),
                ]
            ) + '\n',
            encoding='utf-8',
        )

        stdout = io.StringIO()
        with patch(
            'sys.argv',
            [
                'manga-ocr',
                'build-review-queue',
                '--db',
                str(db_path),
                '--predictions',
                str(predictions_path),
                '--out',
                str(tasks_path),
                '--decisions-out',
                str(decisions_path),
                '--review-strategy',
                'exception-audit',
                '--audit-rate',
                '0.5',
                '--min-audit-pages',
                '1',
                '--audit-seed',
                'fixed-seed',
            ],
        ):
            with redirect_stdout(stdout):
                cli.main()

        payload = json.loads(stdout.getvalue())
        tasks = json.loads(tasks_path.read_text(encoding='utf-8'))
        decisions = read_jsonl(decisions_path)

        self.assertEqual(payload['review_strategy'], 'exception-audit')
        self.assertEqual(payload['review_pages'], 1)
        self.assertEqual(payload['audit_pages'], 1)
        self.assertEqual(payload['auto_accept_pages'], 1)
        self.assertEqual(payload['review_tasks'], 2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(decisions), 3)
        self.assertIn('review', {task['data']['review_route'] for task in tasks})
        self.assertIn('audit', {task['data']['review_route'] for task in tasks})


if __name__ == '__main__':
    unittest.main()
