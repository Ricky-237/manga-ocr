from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import load_train_config


@dataclass(slots=True)
class TrainingStage:
    name: str
    objective: str
    inputs: list[str]
    outputs: list[str]
    entrypoint: str


class OcrTrainingPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_path(cls, config_path: str | Path) -> "OcrTrainingPipeline":
        return cls(load_train_config(config_path))

    def stages(self) -> list[TrainingStage]:
        dataset_manifest = self.config["data"]["dataset_manifest"]
        return [
            TrainingStage(
                name="synthetic-pretrain",
                objective="Bootstrap detector, router, and recognizers on synthetic pages and line crops.",
                inputs=["synthetic generator spec", self.config["teachers"]["detector"]["checkpoint"]],
                outputs=["checkpoints/students/synthetic_pretrain.pt"],
                entrypoint="python -m manga_ocr.train.jobs.synthetic_pretrain",
            ),
            TrainingStage(
                name="silver-gold-finetune",
                objective="Fine-tune students on reviewed gold labels plus filtered silver pseudo-labels.",
                inputs=[dataset_manifest, "manifests/silver_manifest.jsonl"],
                outputs=["checkpoints/students/finetuned.pt"],
                entrypoint="python -m manga_ocr.train.jobs.finetune",
            ),
            TrainingStage(
                name="teacher-distillation",
                objective="Distill teacher logits and teacher detections into mobile students.",
                inputs=[
                    "checkpoints/students/finetuned.pt",
                    self.config["teachers"]["recognizer_jp"]["checkpoint"],
                    self.config["teachers"]["recognizer_kr"]["checkpoint"],
                ],
                outputs=["checkpoints/students/distilled.pt"],
                entrypoint="python -m manga_ocr.train.jobs.distill",
            ),
            TrainingStage(
                name="prune-and-qat",
                objective="Apply structured pruning and quantization-aware training for ONNX INT8 export.",
                inputs=["checkpoints/students/distilled.pt"],
                outputs=["artifacts/mobile/student_bundle_int8.onnx"],
                entrypoint="python -m manga_ocr.train.jobs.quantize",
            ),
        ]

    def benchmark_targets(self) -> dict[str, Any]:
        return dict(self.config["benchmarks"])

    def summary(self) -> dict[str, Any]:
        return {
            "stages": [asdict(stage) for stage in self.stages()],
            "benchmarks": self.benchmark_targets(),
        }
