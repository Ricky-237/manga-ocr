from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..domain import Domain, LabelSource, OcrLineLabel, PolygonPoint, TextDirection, TextType


@dataclass(slots=True)
class TeacherPrediction:
    asset_id: str
    series_id: str
    image_path: str
    domain: Domain
    polygon: list[PolygonPoint]
    transcript: str
    lang: str
    direction: TextDirection
    text_type: TextType
    detection_confidence: float
    recognition_confidence: float
    script_confidence: float
    teacher_agreement: float
    min_text_height: float | None = None

    @property
    def confidence_score(self) -> float:
        return (
            self.detection_confidence
            + self.recognition_confidence
            + self.script_confidence
            + self.teacher_agreement
        ) / 4.0

    def to_line_label(self) -> OcrLineLabel:
        return OcrLineLabel(
            polygon=self.polygon,
            transcript=self.transcript,
            lang=self.lang,
            direction=self.direction,
            text_type=self.text_type,
            confidence=self.confidence_score,
            source=LabelSource.TEACHER,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "series_id": self.series_id,
            "image_path": self.image_path,
            "domain": self.domain.value,
            "polygon": [{"x": point.x, "y": point.y} for point in self.polygon],
            "transcript": self.transcript,
            "lang": self.lang,
            "direction": self.direction.value,
            "text_type": self.text_type.value,
            "detection_confidence": self.detection_confidence,
            "recognition_confidence": self.recognition_confidence,
            "script_confidence": self.script_confidence,
            "teacher_agreement": self.teacher_agreement,
            "min_text_height": self.min_text_height,
            "confidence": self.confidence_score,
        }


@dataclass(slots=True)
class PseudoLabelThresholds:
    min_detection_confidence: float = 0.85
    min_recognition_confidence: float = 0.82
    min_script_confidence: float = 0.80
    min_teacher_agreement: float = 0.75
    min_text_height: float = 12.0


@dataclass(slots=True)
class FilterDecision:
    prediction: TeacherPrediction
    accepted: bool
    reasons: list[str]
    review_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction.to_dict(),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "review_score": self.review_score,
        }


def evaluate_prediction(
    prediction: TeacherPrediction, thresholds: PseudoLabelThresholds | None = None
) -> FilterDecision:
    thresholds = thresholds or PseudoLabelThresholds()
    reasons: list[str] = []

    if prediction.detection_confidence < thresholds.min_detection_confidence:
        reasons.append("low_detection_confidence")
    if prediction.recognition_confidence < thresholds.min_recognition_confidence:
        reasons.append("low_recognition_confidence")
    if prediction.script_confidence < thresholds.min_script_confidence:
        reasons.append("low_script_confidence")
    if prediction.teacher_agreement < thresholds.min_teacher_agreement:
        reasons.append("teacher_disagreement")
    if prediction.min_text_height is not None and prediction.min_text_height < thresholds.min_text_height:
        reasons.append("small_text")

    accepted = not reasons
    review_score = _review_score(prediction, reasons, thresholds)
    return FilterDecision(prediction=prediction, accepted=accepted, reasons=reasons, review_score=review_score)


def filter_for_silver(
    predictions: Sequence[TeacherPrediction], thresholds: PseudoLabelThresholds | None = None
) -> tuple[list[TeacherPrediction], list[FilterDecision]]:
    accepted: list[TeacherPrediction] = []
    rejected: list[FilterDecision] = []
    for prediction in predictions:
        decision = evaluate_prediction(prediction, thresholds=thresholds)
        if decision.accepted:
            accepted.append(prediction)
        else:
            rejected.append(decision)
    return accepted, rejected


def review_queue(
    predictions: Sequence[TeacherPrediction],
    limit: int = 100,
    thresholds: PseudoLabelThresholds | None = None,
) -> list[FilterDecision]:
    decisions = [evaluate_prediction(prediction, thresholds=thresholds) for prediction in predictions]
    decisions.sort(key=lambda item: item.review_score, reverse=True)
    return decisions[:limit]


def _review_score(
    prediction: TeacherPrediction, reasons: Sequence[str], thresholds: PseudoLabelThresholds
) -> float:
    score = 1.0 - prediction.confidence_score
    if prediction.domain == Domain.WEBTOON:
        score += 0.25
    if prediction.direction == TextDirection.VERTICAL:
        score += 0.20
    if prediction.text_type == TextType.SFX:
        score += 0.30
    if prediction.min_text_height is not None and prediction.min_text_height < thresholds.min_text_height:
        score += 0.20
    score += 0.15 * len(reasons)
    return score
