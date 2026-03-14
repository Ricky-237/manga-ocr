from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SyntheticCorpusSpec:
    line_crops: int
    pages: int
    sfx_ratio: float
    render_domains: list[str] = field(default_factory=lambda: ["manga", "webtoon"])
    directions: list[str] = field(default_factory=lambda: ["vertical", "horizontal"])
    augmentations: list[str] = field(
        default_factory=lambda: ["screentone", "moire", "jpeg", "blur", "low_contrast", "mixed_scripts"]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_synthetic_corpus_spec() -> SyntheticCorpusSpec:
    return SyntheticCorpusSpec(
        line_crops=4_000_000,
        pages=400_000,
        sfx_ratio=0.05,
    )
