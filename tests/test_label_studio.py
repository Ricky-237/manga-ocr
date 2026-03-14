from __future__ import annotations

import unittest
from pathlib import Path

from manga_ocr.annotation.label_studio import build_label_studio_tasks, convert_label_studio_export
from manga_ocr.domain import Domain, PageAsset, PolygonPoint, TextDirection, TextType


class LabelStudioTests(unittest.TestCase):
    def test_build_tasks_embeds_prediction_regions(self) -> None:
        asset = PageAsset(
            source_id="api",
            series_id="series-1",
            chapter_id="chapter-1",
            page_index=0,
            image_path="data/raw/abc.png",
            sha256="abc",
            phash="def",
            width=100,
            height=200,
            fetched_at="2026-03-12T00:00:00+00:00",
            domain=Domain.MANGA,
        )

        tasks = build_label_studio_tasks(
            [asset],
            predictions_by_asset={
                asset.asset_id: [
                    {
                        "polygon": [{"x": 10, "y": 20}, {"x": 30, "y": 20}, {"x": 30, "y": 40}],
                        "transcript": "Hello",
                        "lang": "jp",
                        "direction": "vertical",
                        "text_type": "dialogue",
                    }
                ]
            },
        )
        self.assertEqual(len(tasks), 1)
        prediction_results = tasks[0]["predictions"][0]["result"]
        self.assertEqual(len(prediction_results), 4)

    def test_build_tasks_supports_label_studio_local_files_urls(self) -> None:
        asset = PageAsset(
            source_id="api",
            series_id="series-1",
            chapter_id="chapter-1",
            page_index=0,
            image_path="data/raw/abc 1.png",
            sha256="abc",
            phash="def",
            width=100,
            height=200,
            fetched_at="2026-03-12T00:00:00+00:00",
            domain=Domain.MANGA,
        )

        tasks = build_label_studio_tasks(
            [asset],
            local_files_document_root=Path(r"D:\manga-ocr\src"),
        )

        self.assertEqual(tasks[0]["data"]["image"], "/data/local-files/?d=data/raw/abc%201.png")

    def test_convert_export_to_dataset_manifest(self) -> None:
        export_payload = [
            {
                "data": {
                    "image_path": "data/raw/abc.png",
                    "series_id": "series-1",
                    "domain": "webtoon",
                    "width": 100,
                    "height": 200,
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "id": "region-1",
                                "from_name": "text_region",
                                "to_name": "image",
                                "type": "polygonlabels",
                                "original_width": 100,
                                "original_height": 200,
                                "value": {
                                    "points": [[10, 10], [20, 10], [20, 20]],
                                    "polygonlabels": ["caption"],
                                },
                            },
                            {
                                "id": "region-1-text",
                                "parentID": "region-1",
                                "from_name": "transcript",
                                "to_name": "image",
                                "type": "textarea",
                                "value": {"text": ["Bonjour"]},
                            },
                            {
                                "id": "region-1-lang",
                                "parentID": "region-1",
                                "from_name": "language",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["latin"]},
                            },
                            {
                                "id": "region-1-direction",
                                "parentID": "region-1",
                                "from_name": "direction",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["horizontal"]},
                            },
                        ]
                    }
                ],
            }
        ]

        records = convert_label_studio_export(export_payload)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.domain, Domain.WEBTOON)
        self.assertEqual(record.transcript, "Bonjour")
        self.assertEqual(record.direction, TextDirection.HORIZONTAL)
        self.assertEqual(record.text_type, TextType.CAPTION)
        self.assertEqual(record.polygon[0], PolygonPoint(x=10.0, y=20.0))


if __name__ == "__main__":
    unittest.main()
