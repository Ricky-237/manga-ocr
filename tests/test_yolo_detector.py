from __future__ import annotations

import unittest

from manga_ocr.detection.yolo_onnx import YoloOnnxTextDetector, _LetterboxMetadata


class YoloOnnxTextDetectorTests(unittest.TestCase):
    def test_decode_rows_scales_padded_boxes_and_applies_nms(self) -> None:
        detector = YoloOnnxTextDetector(
            model_path='dummy.onnx',
            input_size=1000,
            confidence_threshold=0.50,
            iou_threshold=0.50,
            min_box_size=10.0,
            session=object(),
        )
        metadata = _LetterboxMetadata(
            original_width=800,
            original_height=600,
            square_size=800,
            pad_left=0,
            pad_top=100,
        )
        rows = [
            [100.0, 200.0, 300.0, 400.0, 0.95, 0.0],
            [105.0, 205.0, 295.0, 395.0, 0.80, 0.0],
            [20.0, 110.0, 28.0, 118.0, 0.99, 0.0],
            [400.0, 600.0, 500.0, 700.0, 0.40, 0.0],
        ]

        detections = detector._decode_rows(rows, metadata)

        self.assertEqual(len(detections), 1)
        box = detections[0].box
        self.assertAlmostEqual(box[0], 80.0)
        self.assertAlmostEqual(box[1], 60.0)
        self.assertAlmostEqual(box[2], 240.0)
        self.assertAlmostEqual(box[3], 220.0)
        self.assertEqual(detections[0].label, 'text')

    def test_decode_rows_supports_xywh_boxes(self) -> None:
        detector = YoloOnnxTextDetector(
            model_path='dummy.onnx',
            input_size=1000,
            confidence_threshold=0.50,
            box_format='xywh',
            session=object(),
        )
        metadata = _LetterboxMetadata(
            original_width=1000,
            original_height=1000,
            square_size=1000,
            pad_left=0,
            pad_top=0,
        )

        detections = detector._decode_rows([[500.0, 500.0, 200.0, 100.0, 0.90, 0.0]], metadata)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box, (400.0, 450.0, 600.0, 550.0))


if __name__ == '__main__':
    unittest.main()
