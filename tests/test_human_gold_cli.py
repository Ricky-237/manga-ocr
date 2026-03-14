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
from manga_ocr.splits import SeriesSplitStrategy
from manga_ocr.storage import CatalogDatabase


class HumanGoldCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path('.tmp-tests')
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f'human-gold-{uuid4().hex}'
        self.workdir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _insert_asset(self, db_path: Path, series_id: str, chapter_id: str, page_index: int) -> PageAsset:
        catalog = CatalogDatabase(db_path)
        catalog.initialize()
        asset = PageAsset(
            source_id='consumet-mangahere',
            series_id=series_id,
            chapter_id=chapter_id,
            page_index=page_index,
            image_path=str(self.workdir / f'{series_id}-{page_index}.jpg'),
            sha256=f'sha-{series_id}-{page_index}',
            phash=None,
            width=800,
            height=1200,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            domain=Domain.MANGA,
            metadata={},
        )
        catalog.upsert_page_asset(asset)
        return asset

    def test_prepare_human_gold_exports_tasks_and_manifest(self) -> None:
        from manga_ocr import cli

        db_path = self.workdir / 'catalog.sqlite'
        tasks_path = self.workdir / 'human_gold_tasks.json'
        manifest_path = self.workdir / 'human_gold_manifest.jsonl'
        predictions_path = self.workdir / 'teacher_predictions.jsonl'

        asset_a = self._insert_asset(db_path, series_id='series-a', chapter_id='series-a/c1', page_index=0)
        asset_b = self._insert_asset(db_path, series_id='series-b', chapter_id='series-b/c1', page_index=0)
        asset_c = self._insert_asset(db_path, series_id='series-c', chapter_id='series-c/c1', page_index=0)

        predictions_path.write_text(
            '\n'.join(
                [
                    json.dumps(
                        {
                            'asset_id': asset_a.asset_id,
                            'series_id': asset_a.series_id,
                            'image_path': asset_a.image_path,
                            'domain': asset_a.domain.value,
                            'polygon': [
                                {'x': 0, 'y': 0},
                                {'x': 48, 'y': 0},
                                {'x': 48, 'y': 20},
                                {'x': 0, 'y': 20},
                            ],
                            'transcript': 'easy',
                            'lang': 'ja',
                            'direction': 'vertical',
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
                            'polygon': [
                                {'x': 10, 'y': 10},
                                {'x': 56, 'y': 10},
                                {'x': 56, 'y': 28},
                                {'x': 10, 'y': 28},
                            ],
                            'transcript': 'hard',
                            'lang': 'ja',
                            'direction': 'vertical',
                            'text_type': 'dialogue',
                            'detection_confidence': 0.40,
                            'recognition_confidence': 0.45,
                            'script_confidence': 0.42,
                            'teacher_agreement': 0.30,
                            'min_text_height': 6,
                        }
                    ),
                ]
            )
            + '\n',
            encoding='utf-8',
        )

        stdout = io.StringIO()
        with patch(
            'sys.argv',
            [
                'manga-ocr',
                'prepare-human-gold',
                '--db',
                str(db_path),
                '--predictions',
                str(predictions_path),
                '--tasks-out',
                str(tasks_path),
                '--manifest-out',
                str(manifest_path),
                '--page-limit',
                '3',
                '--image-base-url',
                'https://labels.example/assets',
            ],
        ):
            with redirect_stdout(stdout):
                cli.main()

        payload = json.loads(stdout.getvalue())
        tasks = json.loads(tasks_path.read_text(encoding='utf-8'))
        manifest_records = read_jsonl(manifest_path)
        split_strategy = SeriesSplitStrategy()

        self.assertEqual(payload['selected_pages'], 3)
        self.assertEqual(payload['preannotated_pages'], 2)
        self.assertEqual(payload['tasks_output'], str(tasks_path))
        self.assertEqual(payload['manifest_output'], str(manifest_path))
        self.assertEqual(len(tasks), 3)
        self.assertEqual(len(manifest_records), 3)

        task_by_asset_id = {task['data']['asset_id']: task for task in tasks}
        self.assertIn(asset_a.asset_id, task_by_asset_id)
        self.assertIn(asset_b.asset_id, task_by_asset_id)
        self.assertIn(asset_c.asset_id, task_by_asset_id)

        for asset in (asset_a, asset_b, asset_c):
            task = task_by_asset_id[asset.asset_id]
            expected_split = split_strategy.assign(asset.series_id).value
            self.assertEqual(task['data']['target_split'], expected_split)
            expected_review_mode = 'double_review' if expected_split == 'test' else 'single_review'
            self.assertEqual(task['data']['review_mode'], expected_review_mode)
            self.assertTrue(task['data']['image'].startswith('https://labels.example/assets/'))

        self.assertIn('predictions', task_by_asset_id[asset_a.asset_id])
        self.assertIn('predictions', task_by_asset_id[asset_b.asset_id])
        self.assertNotIn('predictions', task_by_asset_id[asset_c.asset_id])
        self.assertGreater(task_by_asset_id[asset_b.asset_id]['data']['review_priority'], 0.0)


    def test_prepare_human_gold_supports_local_files_document_root(self) -> None:
        from manga_ocr import cli

        db_path = self.workdir / 'catalog.sqlite'
        tasks_path = self.workdir / 'human_gold_local_files.json'
        manifest_path = self.workdir / 'human_gold_local_files_manifest.jsonl'

        self._insert_asset(db_path, series_id='series-a', chapter_id='series-a/c1', page_index=0)
        stdout = io.StringIO()
        with patch(
            'sys.argv',
            [
                'manga-ocr',
                'prepare-human-gold',
                '--db',
                str(db_path),
                '--tasks-out',
                str(tasks_path),
                '--manifest-out',
                str(manifest_path),
                '--page-limit',
                '1',
                '--local-files-document-root',
                str(self.workdir),
            ],
        ):
            with redirect_stdout(stdout):
                cli.main()

        tasks = json.loads(tasks_path.read_text(encoding='utf-8'))
        self.assertEqual(
            tasks[0]['data']['image'],
            f"/data/local-files/?d={(self.workdir / 'series-a-0.jpg').as_posix()}",
        )



if __name__ == '__main__':
    unittest.main()
