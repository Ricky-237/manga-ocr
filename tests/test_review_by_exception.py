from __future__ import annotations

import unittest
from pathlib import Path

from manga_ocr.domain import Domain, PageAsset, PolygonPoint, TextDirection, TextType
from manga_ocr.pseudo_label import (
    ExceptionReviewPolicy,
    TeacherPrediction,
    build_review_by_exception_tasks,
    summarize_exception_review,
)


class ReviewByExceptionTests(unittest.TestCase):
    def _asset(self, suffix: str) -> PageAsset:
        return PageAsset(
            source_id='api',
            series_id=f'series-{suffix}',
            chapter_id=f'series-{suffix}/c1',
            page_index=0,
            image_path=f'data/raw/{suffix}.png',
            sha256=f'sha-{suffix}',
            phash=None,
            width=100,
            height=200,
            fetched_at='2026-03-14T00:00:00+00:00',
            domain=Domain.MANGA,
        )

    def _prediction(
        self,
        asset: PageAsset,
        transcript: str,
        *,
        detection: float = 0.97,
        recognition: float = 0.96,
        script: float = 0.95,
        agreement: float = 0.94,
        min_height: float = 18.0,
    ) -> TeacherPrediction:
        return TeacherPrediction(
            asset_id=asset.asset_id,
            series_id=asset.series_id,
            image_path=asset.image_path,
            domain=asset.domain,
            polygon=[PolygonPoint(0, 0), PolygonPoint(10, 0), PolygonPoint(10, 10)],
            transcript=transcript,
            lang='jp',
            direction=TextDirection.HORIZONTAL,
            text_type=TextType.DIALOGUE,
            detection_confidence=detection,
            recognition_confidence=recognition,
            script_confidence=script,
            teacher_agreement=agreement,
            min_text_height=min_height,
        )

    def test_build_review_by_exception_tasks_reviews_failures_and_samples_audit(self) -> None:
        auto_a = self._asset('a')
        auto_b = self._asset('b')
        hard = self._asset('hard')
        empty = self._asset('empty')
        predictions = [
            self._prediction(auto_a, 'easy-a'),
            self._prediction(auto_b, 'easy-b'),
            self._prediction(hard, 'tiny', detection=0.4, recognition=0.45, script=0.42, agreement=0.30, min_height=6),
        ]

        tasks, decisions = build_review_by_exception_tasks(
            [auto_a, auto_b, hard, empty],
            predictions,
            policy=ExceptionReviewPolicy(audit_rate=0.5, min_audit_pages=1, audit_seed='fixed-seed'),
            local_files_document_root=Path(r'D:\manga-ocr\src'),
        )

        route_by_asset = {decision.asset.asset_id: decision.route for decision in decisions}
        self.assertEqual(route_by_asset[hard.asset_id], 'review')
        self.assertEqual(route_by_asset[empty.asset_id], 'review')
        self.assertEqual(sum(1 for route in route_by_asset.values() if route == 'audit'), 1)
        self.assertEqual(sum(1 for route in route_by_asset.values() if route == 'auto_accept'), 1)

        task_routes = {task['data']['asset_id']: task['data']['review_route'] for task in tasks}
        self.assertEqual(task_routes[hard.asset_id], 'review')
        self.assertEqual(task_routes[empty.asset_id], 'review')
        self.assertIn('/data/local-files/?d=data/raw/', tasks[0]['data']['image'])

        summary = summarize_exception_review(decisions)
        self.assertEqual(summary['review_pages'], 2)
        self.assertEqual(summary['audit_pages'], 1)
        self.assertEqual(summary['auto_accept_pages'], 1)
        self.assertIn('low_detection_confidence', summary['review_reason_counts'])
        self.assertIn('no_teacher_predictions', summary['review_reason_counts'])
