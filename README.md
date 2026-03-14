# Manga OCR Pipeline

This repository bootstraps an OCR training stack for manga and webtoons where chapter images arrive from an API without annotations. The implementation is organized around a data-first loop:

`ingestion -> durable storage -> pseudo-labeling -> targeted human review -> gold/silver manifests -> training plan -> Android SDK`

## What is implemented

- A stable `ChapterApiAdapter` contract with an HTTP base adapter.
- A concrete `ConsumetMangahereAdapter` wired to the `/latest`, `/info`, and `/read` endpoints you provided.
- Ready-to-use CLI shortcuts to ingest the latest page or a specific manga without manually supplying chapter ids.
- Durable storage for downloaded pages with `sha256`, perceptual hash, image metadata, and duplicate tracking in SQLite.
- Deterministic dataset splitting by `series_id` to avoid leakage between train/val/test.
- Manifest writers for `page_assets.jsonl` and `dataset_manifest.jsonl`.
- Label Studio task generation plus export conversion into the canonical dataset manifest.
- Teacher prediction import from JSON, JSONL, or `{ "predictions": [...] }` payloads.
- A one-shot `run-teacher-cycle` command that writes teacher predictions, a silver manifest, and a prioritized review queue in one pass.
- A reusable YOLO ONNX text detector plus a composite `YOLO detector -> crop recognizer` teacher backend for manga/webtoon pages.
- An `evaluate-predictions` command that compares predictions against a reviewed gold manifest and reports recall, precision, CER, and WER by slice.
- Silver filtering plus review-queue export back into Label Studio tasks.
- A training-plan scaffold around the four-stage student pipeline plus benchmark gates.
- An Android SDK skeleton centered on a single `runPage(bitmap)` API.

## Repository layout

- `src/manga_ocr/ingest`: API adapters and ingestion pipeline.
- `src/manga_ocr/manifests`: JSONL manifest helpers.
- src/manga_ocr/pseudo_label: teacher prediction I/O, silver filtering, and review prioritization.
- `src/manga_ocr/detection`: reusable detector backends, including YOLO ONNX text detection.
- `src/manga_ocr/colab`: Colab-oriented dataset preparation workflow and bundle helpers.
- src/manga_ocr/teachers: pluggable teacher backends, including a YOLO26N+Tesseract teacher and a Tesseract-only fallback.
- `src/manga_ocr/annotation`: Label Studio configuration and converters.
- `src/manga_ocr/train`: config loading and training-stage planning.
- `android-sdk`: Android integration scaffold for mobile inference.
- configs/train_ocr.yaml: canonical training configuration template.
- configs/teacher_yolo26_tesseract.json: sample predictor config for the built-in YOLO26N detector + Tesseract recognizer teacher.
- configs/teacher_tesseract.json: sample predictor config for the built-in Tesseract teacher.
- scripts/prepare_colab_dataset.py: one-command Colab/local dataset preparation entrypoint.
- notebooks/manga_ocr_dataset_prep_colab.ipynb: ready-to-run Colab notebook from API download to dataset bundle export.

## Consumet Mangahere adapter

The adapter `manga_ocr.ingest.consumet_mangahere:ConsumetMangahereAdapter` now targets:

- `GET /latest?page={page}` to enumerate the latest manga page.
- `GET /info?id={id}` to resolve chapters for a manga.
- `GET /read?chapterId={id}` to enumerate page images for a chapter.

All JSON requests automatically send `Referer: https://mangahere.com/` plus browser-like headers. Image downloads now try `Referer: https://mangahere.com/` first, then retry once with the per-image `headerForImage` referer returned by `/read` if the CDN answers `403`. The adapter also uses retries for transient transport failures and a dedicated image timeout that defaults to 90 seconds, which is much safer for slow CDN responses than the metadata timeout.

## Quick start

1. Create the catalog database:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli init-db --db data/catalog.sqlite
```

2. Ingest a single known chapter:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli ingest `
  --adapter manga_ocr.ingest.consumet_mangahere:ConsumetMangahereAdapter `
  --db data/catalog.sqlite `
  --storage-root data `
  --chapter-id one_piece/v98/c1176
```

3. Ingest the latest chapters returned by a latest-page feed:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli ingest-latest `
  --db data/catalog.sqlite `
  --storage-root data `
  --page 1 `
  --limit 10 `
  --image-timeout-s 120 `
  --max-retries 3 `
  --continue-on-error
```

4. Ingest the newest chapter for a specific manga:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli ingest-manga `
  --db data/catalog.sqlite `
  --storage-root data `
  --manga-id one_piece `
  --image-timeout-s 120 `
  --max-retries 3 `
  --continue-on-error
```

5. Ingest all chapters for a specific manga:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli ingest-manga `
  --db data/catalog.sqlite `
  --storage-root data `
  --manga-id one_piece `
  --image-timeout-s 120 `
  --max-retries 3 `
  --continue-on-error `
  --all-chapters
```

6. Export Label Studio tasks for fully manual labeling if needed:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli build-label-studio-tasks `
  --db data/catalog.sqlite `
  --out annotation/tasks/chapter_tasks.json
```

7. Run the YOLO26N detector-backed teacher over ingested assets and export canonical predictions:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli run-teacher `
  --db data/catalog.sqlite `
  --out manifests/teacher_predictions.jsonl `
  --predictor manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor `
  --predictor-config configs/teacher_yolo26_tesseract.json `
  --series-id one_piece `
  --continue-on-error
```

8. Or run the whole detector-to-teacher-to-silver-to-review cycle in one command:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli run-teacher-cycle `
  --db data/catalog.sqlite `
  --predictor manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor `
  --predictor-config configs/teacher_yolo26_tesseract.json `
  --predictions-out manifests/teacher_predictions.jsonl `
  --silver-out manifests/silver_manifest.jsonl `
  --review-out annotation/tasks/review_queue.json `
  --rejected-out manifests/rejected_predictions.jsonl `
  --review-decisions-out manifests/review_decisions.jsonl `
  --limit 200 `
  --continue-on-error
```

If you still want a lightweight OCR-only fallback with no detector model, keep using `manga_ocr.teachers.tesseract:TesseractTeacherPredictor` with `configs/teacher_tesseract.json`.

9. Build a silver manifest from teacher predictions:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli build-silver-manifest `
  --input manifests/teacher_predictions.jsonl `
  --db data/catalog.sqlite `
  --out manifests/silver_manifest.jsonl `
  --rejected-out manifests/rejected_predictions.jsonl
```

10. Export a prioritized review queue with teacher pre-annotations:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli build-review-queue `
  --db data/catalog.sqlite `
  --predictions manifests/teacher_predictions.jsonl `
  --out annotation/tasks/review_queue.json `
  --page-limit 200
```

For a review-by-exception workflow with automatic acceptance of confident lines plus deterministic audit sampling, use:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli build-review-queue `
  --db data/catalog.sqlite `
  --predictions manifests/teacher_predictions.jsonl `
  --out annotation/tasks/exception_review.json `
  --decisions-out manifests/exception_review_decisions.jsonl `
  --review-strategy exception-audit `
  --audit-rate 0.02 `
  --min-audit-pages 10 `
  --local-files-document-root D:\manga-ocr\src
```

This strategy sends only risky pages and a small audit sample of auto-accepted pages to Label Studio.

11. Prepare a human gold batch with stable train/val/test assignment and double review on the test split:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli prepare-human-gold `
  --db data/catalog.sqlite `
  --predictions manifests/teacher_predictions.jsonl `
  --tasks-out annotation/tasks/human_gold_batch.json `
  --manifest-out manifests/human_gold_batch.jsonl `
  --page-limit 2500 `
  --local-files-document-root D:\manga-ocr\src
```

The exported Label Studio tasks include:

- `target_split` to keep the gold set split stable by `series_id`
- `review_mode` set to `double_review` for test pages and `single_review` otherwise
- optional teacher pre-annotations when `--predictions` is provided
- `/data/local-files/?d=...` image URLs when `--local-files-document-root` is set

For Label Studio local files mode, configure:

- `LOCAL_FILES_SERVING_ENABLED=true`
- `LOCAL_FILES_DOCUMENT_ROOT=D:\manga-ocr\src`
- in the Label Studio UI, create a `Local Files` source storage whose `Absolute local path` is a subdirectory such as `D:\manga-ocr\src\data`

12. Convert reviewed annotations to the canonical manifest:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli convert-label-studio-export `
  --input annotation/exports/reviewed.json `
  --out manifests/dataset_manifest.jsonl
```

13. Evaluate predictions against a reviewed gold manifest:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli evaluate-predictions `
  --gold manifests/dataset_manifest.jsonl `
  --predictions manifests/teacher_predictions.jsonl `
  --out reports/teacher_eval.json
```

14. Inspect the training plan:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli plan-training --config configs/train_ocr.yaml
```

## Colab dataset prep

A Colab-ready workflow is now included for the full path `download -> ingest -> page_assets -> teacher_predictions -> silver_manifest -> dataset zip`.

Notebook:

- [manga_ocr_dataset_prep_colab.ipynb](D:/manga-ocr/notebooks/manga_ocr_dataset_prep_colab.ipynb)

Script:

```powershell
python scripts/prepare_colab_dataset.py `
  --workspace-root /content/manga-ocr-workdir `
  --manga-id one_piece `
  --chapter-limit 3 `
  --detector-model-path /content/drive/MyDrive/models/yolo26n.onnx
```

The script always exports `data/catalog.sqlite`, `manifests/page_assets.jsonl`, raw images under `data/raw`, and when the teacher is enabled it also writes `teacher_predictions.jsonl`, `silver_manifest.jsonl`, `rejected_predictions.jsonl`, `review_queue.json`, and a final `dataset_bundle.zip`.

## Teacher prediction schema

Each record must contain at least:

- `asset_id`
- `polygon`
- `transcript`

If `--db` is not provided, each record must also include `series_id`, `image_path`, and `domain`.

Example JSONL record:

```json
{"asset_id":"api:series-1:chapter-1:0000","series_id":"series-1","image_path":"data/raw/ab/abcdef.png","domain":"manga","polygon":[{"x":12,"y":20},{"x":60,"y":20},{"x":60,"y":48},{"x":12,"y":48}],"transcript":"Bonjour","lang":"latin","direction":"horizontal","text_type":"dialogue","detection_confidence":0.96,"recognition_confidence":0.94,"script_confidence":0.93,"teacher_agreement":0.91,"min_text_height":18}
```

## Dependencies

The repository keeps runtime requirements minimal and uses optional integrations:

- `Pillow` and `numpy` for perceptual hashing.
- `PyYAML` for the training config loader.
- `torch` for real model/training code when you add the student and teacher implementations.
- `onnxruntime`, `numpy`, and `Pillow` for the built-in YOLO26N detector-backed teacher.

The current scaffold and tests run with the Python standard library, except for `plan-training`, which still requires the declared `train` extra because it loads YAML.



## Teacher predictor contract

`run-teacher` loads a Python class from `--predictor package.module:ClassName`. The class must expose:

```python
from manga_ocr.domain import PageAsset

class MyTeacherPredictor:
    def predict_page(self, asset: PageAsset):
        return [
            {
                "polygon": [{"x": 0, "y": 0}, {"x": 32, "y": 0}, {"x": 32, "y": 16}, {"x": 0, "y": 16}],
                "transcript": "こんにちは",
                "lang": "ja",
                "direction": "vertical",
                "text_type": "dialogue",
                "detection_confidence": 0.97,
                "recognition_confidence": 0.96,
                "script_confidence": 0.95,
                "teacher_agreement": 0.94,
            }
        ]
```

The predictor can return either canonical `TeacherPrediction` objects or plain dictionaries. Missing `asset_id`, `series_id`, `image_path`, and `domain` are filled automatically from the selected `PageAsset`.
## Built-in YOLO26N detector teacher

The repository now includes `manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor`, a composite teacher that uses your fine-tuned YOLO26N ONNX model for text detection and Tesseract only for crop-level recognition.

Use the bundled config file [configs/teacher_yolo26_tesseract.json](D:/manga-ocr/configs/teacher_yolo26_tesseract.json) and point `detector_model_path` to your ONNX file. The sample config is already aligned with `D:/engine-fault-detection/manga109_yolo/yolo26n.onnx`.

Example:

```powershell
$env:PYTHONPATH = "src"
python -m manga_ocr.cli run-teacher `
  --db data/catalog.sqlite `
  --out manifests/teacher_predictions.jsonl `
  --predictor manga_ocr.teachers.yolo_tesseract:YoloOnnxTesseractTeacherPredictor `
  --predictor-config configs/teacher_yolo26_tesseract.json `
  --series-id one_piece `
  --continue-on-error
```

This keeps the existing pseudo-label pipeline unchanged while upgrading the detection stage to a manga/webtoon-specialized model.

## Tesseract-only fallback

The original `manga_ocr.teachers.tesseract:TesseractTeacherPredictor` is still available as a lightweight fallback when you want to validate the OCR loop without the ONNX detector.

It requires a local Tesseract installation with the language packs you want to use, for example `jpn`, `kor`, and `eng`. The bundled config file [configs/teacher_tesseract.json](D:/manga-ocr/configs/teacher_tesseract.json) remains available for that mode.







