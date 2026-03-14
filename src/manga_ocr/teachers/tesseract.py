from __future__ import annotations

import csv
import io
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

from ..domain import PageAsset, PolygonPoint, TextDirection, TextType
from ..pseudo_label.teacher import TeacherPredictor


@dataclass(slots=True)
class _WordBox:
    left: int
    top: int
    width: int
    height: int
    confidence: float
    text: str
    line_key: tuple[int, int, int, int]


class TesseractTeacherPredictor(TeacherPredictor):
    def __init__(
        self,
        executable: str = "tesseract",
        languages: str = "jpn+kor+eng",
        psm: int = 11,
        oem: int = 1,
        default_direction: str = "horizontal",
        default_text_type: str = "dialogue",
        default_lang: str = "mixed",
        min_word_confidence: float = 35.0,
        min_line_confidence: float = 0.45,
        min_text_height: float = 8.0,
        min_transcript_length: int = 1,
        preserve_spaces: bool = True,
        timeout_s: float = 120.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.executable = executable
        self.languages = languages
        self.psm = psm
        self.oem = oem
        self.default_direction = TextDirection(default_direction)
        self.default_text_type = TextType(default_text_type)
        self.default_lang = default_lang
        self.min_word_confidence = min_word_confidence
        self.min_line_confidence = min_line_confidence
        self.min_text_height = min_text_height
        self.min_transcript_length = max(min_transcript_length, 1)
        self.preserve_spaces = preserve_spaces
        self.timeout_s = timeout_s
        self.extra_args = list(extra_args or [])

    def predict_page(self, asset: PageAsset):
        command = self._build_command(asset.image_path)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Tesseract executable not found: {self.executable!r}. Install Tesseract OCR and/or set 'executable'."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Tesseract timed out after {self.timeout_s}s while processing {asset.image_path}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(f"Tesseract failed for {asset.image_path}: {message}")

        return self._parse_tsv(result.stdout or "", asset)

    def _build_command(self, image_path: str | Path) -> list[str]:
        command = [
            self.executable,
            str(image_path),
            "stdout",
            "-l",
            self.languages,
            "--oem",
            str(self.oem),
            "--psm",
            str(self.psm),
        ]
        command.extend(self.extra_args)
        command.append("tsv")
        return command

    def _parse_tsv(self, tsv_text: str, asset: PageAsset) -> list[dict[str, object]]:
        words = [word for word in self._iter_word_boxes(tsv_text) if word.text]
        by_line: dict[tuple[int, int, int, int], list[_WordBox]] = defaultdict(list)
        for word in words:
            by_line[word.line_key].append(word)

        predictions: list[dict[str, object]] = []
        for line_words in by_line.values():
            transcript = self._join_tokens(word.text for word in line_words)
            if len(transcript.strip()) < self.min_transcript_length:
                continue

            line_confidence = max(0.0, min(1.0, fmean(word.confidence for word in line_words) / 100.0))
            box_height = max(word.top + word.height for word in line_words) - min(word.top for word in line_words)
            if line_confidence < self.min_line_confidence:
                continue
            if box_height < self.min_text_height:
                continue

            predictions.append(
                {
                    "asset_id": asset.asset_id,
                    "series_id": asset.series_id,
                    "image_path": asset.image_path,
                    "domain": asset.domain.value,
                    "polygon": self._line_polygon(line_words),
                    "transcript": transcript,
                    "lang": self._infer_lang(transcript),
                    "direction": self.default_direction.value,
                    "text_type": self.default_text_type.value,
                    "detection_confidence": line_confidence,
                    "recognition_confidence": line_confidence,
                    "script_confidence": self._estimate_script_confidence(transcript),
                    "teacher_agreement": line_confidence,
                    "min_text_height": float(box_height),
                }
            )
        return predictions

    def _iter_word_boxes(self, tsv_text: str) -> Iterable[_WordBox]:
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        for row in reader:
            text = str(row.get("text") or "").strip()
            conf = self._parse_float(row.get("conf"), default=-1.0)
            if conf < self.min_word_confidence:
                continue
            width = self._parse_int(row.get("width"))
            height = self._parse_int(row.get("height"))
            if width <= 0 or height <= 0:
                continue
            yield _WordBox(
                left=self._parse_int(row.get("left")),
                top=self._parse_int(row.get("top")),
                width=width,
                height=height,
                confidence=conf,
                text=text,
                line_key=(
                    self._parse_int(row.get("page_num"), default=0),
                    self._parse_int(row.get("block_num"), default=0),
                    self._parse_int(row.get("par_num"), default=0),
                    self._parse_int(row.get("line_num"), default=0),
                ),
            )

    def _join_tokens(self, tokens: Iterable[str]) -> str:
        clean_tokens = [token.strip() for token in tokens if token and token.strip()]
        if self.preserve_spaces:
            return " ".join(clean_tokens)
        return "".join(clean_tokens)

    @staticmethod
    def _line_polygon(words: list[_WordBox]) -> list[dict[str, float]]:
        left = min(word.left for word in words)
        top = min(word.top for word in words)
        right = max(word.left + word.width for word in words)
        bottom = max(word.top + word.height for word in words)
        return [
            {"x": float(left), "y": float(top)},
            {"x": float(right), "y": float(top)},
            {"x": float(right), "y": float(bottom)},
            {"x": float(left), "y": float(bottom)},
        ]

    def _infer_lang(self, transcript: str) -> str:
        if self.default_lang != "mixed":
            return self.default_lang
        has_hangul = any("\uac00" <= char <= "\ud7af" for char in transcript)
        has_hiragana = any("\u3040" <= char <= "\u309f" for char in transcript)
        has_katakana = any("\u30a0" <= char <= "\u30ff" for char in transcript)
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in transcript)
        has_latin = any(char.isascii() and char.isalpha() for char in transcript)

        if has_hangul and not (has_hiragana or has_katakana):
            return "ko"
        if has_hiragana or has_katakana:
            return "ja"
        if has_cjk and not has_hangul:
            return "ja"
        if has_latin and not (has_hangul or has_hiragana or has_katakana or has_cjk):
            return "latin"
        return "mixed"

    def _estimate_script_confidence(self, transcript: str) -> float:
        if not transcript.strip():
            return 0.0
        lang = self._infer_lang(transcript)
        if lang == "mixed":
            return 0.85
        return 0.95

    @staticmethod
    def _parse_int(value: object, default: int = 0) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(value: object, default: float = 0.0) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default
