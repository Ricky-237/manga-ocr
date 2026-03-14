from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from manga_ocr.colab.dataset_prep import (
    ColabDatasetWorkspace,
    _build_dataset_archive,
    _resolve_predictor_config,
)


class ColabDatasetPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_root = Path('.tmp-tests')
        self.scratch_root.mkdir(exist_ok=True)
        self.workdir = self.scratch_root / f'colab-{uuid4().hex}'
        self.workdir.mkdir(parents=True)
        self.workspace = ColabDatasetWorkspace.from_root(self.workdir / 'workspace')
        self.workspace.initialize()

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_resolve_predictor_config_writes_workspace_config_from_detector_path(self) -> None:
        model_path = self.workdir / 'models' / 'yolo26n.onnx'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b'onnx-model-placeholder')

        config_path = _resolve_predictor_config(
            workspace=self.workspace,
            predictor_config_path=None,
            detector_model_path=model_path,
        )

        payload = json.loads(config_path.read_text(encoding='utf-8'))
        self.assertEqual(config_path, self.workspace.predictor_config_path)
        self.assertEqual(payload['detector_model_path'], str(model_path))
        self.assertIn('recognizer', payload)

    def test_build_dataset_archive_keeps_workspace_and_external_config_files(self) -> None:
        self.workspace.db_path.write_text('db', encoding='utf-8')
        self.workspace.page_assets_manifest.write_text('{}\n', encoding='utf-8')
        self.workspace.summary_path.write_text('{}', encoding='utf-8')
        sample_raw = self.workspace.raw_dir / 'ab' / 'file.jpg'
        sample_raw.parent.mkdir(parents=True, exist_ok=True)
        sample_raw.write_bytes(b'raw')

        external_config = self.workdir / 'external_config.json'
        external_config.write_text('{}', encoding='utf-8')
        archive_path = self.workspace.artifacts_dir / 'bundle.zip'

        _build_dataset_archive(
            workspace=self.workspace,
            output_path=archive_path,
            include_teacher_outputs=False,
            predictor_config_path=external_config,
        )

        import zipfile

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())

        self.assertIn('data/catalog.sqlite', names)
        self.assertIn('manifests/page_assets.jsonl', names)
        self.assertIn('artifacts/dataset_summary.json', names)
        self.assertIn('data/raw/ab/file.jpg', names)
        self.assertIn('external/external_config.json', names)


if __name__ == '__main__':
    unittest.main()
