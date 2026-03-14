from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import PageAsset
from ..manifests import write_jsonl
from .filtering import TeacherPrediction
from .io import teacher_prediction_from_dict

PredictionLike = TeacherPrediction | Mapping[str, Any]


class TeacherPredictor(ABC):
    @abstractmethod
    def predict_page(self, asset: PageAsset) -> Sequence[PredictionLike]:
        raise NotImplementedError

    def predict_pages(self, assets: Sequence[PageAsset]) -> list[TeacherPrediction]:
        predictions, failures = run_teacher_predictions(self, assets, continue_on_error=False)
        if failures:
            raise RuntimeError(f"Teacher prediction failed for {len(failures)} asset(s)")
        return predictions


@dataclass(slots=True)
class TeacherAssetFailure:
    asset_id: str
    image_path: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "image_path": self.image_path,
            "error": self.error,
        }


def load_teacher_predictor(import_path: str, config_path: str | None = None) -> TeacherPredictor:
    module_name, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    predictor_class = getattr(module, class_name)

    config: dict[str, Any] = {}
    if config_path:
        resolved_path = _resolve_config_path(config_path)
        config = json.loads(resolved_path.read_text(encoding="utf-8"))

    predictor = predictor_class(**config)
    if not hasattr(predictor, "predict_page"):
        raise TypeError(f"{import_path} does not expose predict_page(asset)")
    return predictor


def select_page_assets(
    page_assets: Sequence[PageAsset],
    source_id: str | None = None,
    series_id: str | None = None,
    chapter_id: str | None = None,
    limit: int | None = None,
    include_duplicates: bool = False,
) -> list[PageAsset]:
    selected: list[PageAsset] = []
    for asset in page_assets:
        if not include_duplicates and asset.is_duplicate:
            continue
        if source_id and asset.source_id != source_id:
            continue
        if series_id and asset.series_id != series_id:
            continue
        if chapter_id and asset.chapter_id != chapter_id:
            continue
        selected.append(asset)
        if limit is not None and len(selected) >= max(limit, 0):
            break
    return selected


def run_teacher_predictions(
    predictor: TeacherPredictor,
    page_assets: Sequence[PageAsset],
    continue_on_error: bool = True,
) -> tuple[list[TeacherPrediction], list[TeacherAssetFailure]]:
    predictions: list[TeacherPrediction] = []
    failures: list[TeacherAssetFailure] = []

    for asset in page_assets:
        try:
            raw_predictions = predictor.predict_page(asset)
            normalized = _normalize_predictions(raw_predictions, asset)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(TeacherAssetFailure(asset_id=asset.asset_id, image_path=asset.image_path, error=str(exc)))
            continue
        predictions.extend(normalized)

    return predictions, failures


def write_teacher_predictions(predictions: Sequence[TeacherPrediction], output_path: str | Path) -> None:
    write_jsonl((prediction.to_dict() for prediction in predictions), output_path)


def _normalize_predictions(
    raw_predictions: Sequence[PredictionLike] | None,
    asset: PageAsset,
) -> list[TeacherPrediction]:
    if raw_predictions is None:
        return []

    normalized: list[TeacherPrediction] = []
    asset_lookup = {asset.asset_id: asset}
    for raw_prediction in raw_predictions:
        if isinstance(raw_prediction, TeacherPrediction):
            normalized.append(raw_prediction)
            continue
        payload = dict(raw_prediction)
        payload.setdefault("asset_id", asset.asset_id)
        payload.setdefault("series_id", asset.series_id)
        payload.setdefault("image_path", asset.image_path)
        payload.setdefault("domain", asset.domain.value)
        payload.setdefault("lang", "mixed")
        payload.setdefault("direction", "horizontal")
        payload.setdefault("text_type", "unknown")
        normalized.append(teacher_prediction_from_dict(payload, page_assets_by_id=asset_lookup))
    return normalized


def _resolve_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path)
    candidates = [candidate]
    if not candidate.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / candidate)

    for path_candidate in candidates:
        if path_candidate.exists():
            return path_candidate

    tried = ", ".join(str(path_candidate) for path_candidate in candidates)
    raise FileNotFoundError(f"Could not find predictor config '{config_path}'. Tried: {tried}")
