# Android SDK scaffold

This module is a thin Android-facing wrapper around the future ONNX Runtime Mobile inference bundle. It intentionally exposes a single entry point:

`runPage(bitmap): List<OcrBlock>`

## Current state

- Data models for polygon points and OCR blocks are defined.
- The public `MangaOcrEngine` interface is stable.
- `OnnxMangaOcrEngine` is a placeholder for the future ONNX Runtime Mobile implementation with NNAPI-first execution.

## Expected next steps

1. Add the ONNX Runtime Mobile dependency and a packaged student bundle.
2. Implement tiling for long webtoons and page normalization to `1280px` long side.
3. Add deterministic reading-order resolution before returning `OcrBlock` results.
