"""
YAML experiment configs → a typed :class:`ExperimentConfig`.

Every CLI in ``scripts/`` takes ``--config configs/<name>.yaml``. Keys not
listed here are kept in :attr:`ExperimentConfig.extra` so a config can carry
method-specific knobs without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .tokens import parse_item


@dataclass
class GenerationConfig:
    seed: int = 42
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 1.0

    def kwargs(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }


@dataclass
class ExperimentConfig:
    name: str
    prompts: List[str]
    items: List[str]
    easy_count: int
    fallback_word: Optional[str] = None       # category token for items without their own token
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    max_sequence_length: int = 512
    position: str = "last"                    # which sub-token of a multi-piece item to read
    layer_stride: int = 3                     # diffusion lens: sample every k-th depth
    topk: int = 5
    prob_thresh: float = 0.05                 # L* thresholds
    rank_thresh: int = 50
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    results_dir: Path = Path("results")
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def out(self) -> Path:
        return Path(self.results_dir)

    def lens_dir(self, lens: str) -> Path:
        d = self.out / lens / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "ExperimentConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})

        prompts = list(raw.pop("prompts"))
        items = raw.pop("items", None) or [parse_item(p) for p in prompts]
        easy_count = raw.pop("easy_count", len(prompts) // 2)
        gen = GenerationConfig(**raw.pop("generation", {}))
        results_dir = Path(raw.pop("results_dir", "results"))
        name = raw.pop("name", Path(path).stem)

        known = {f.name for f in cls.__dataclass_fields__.values()} - {"extra"}
        fields = {k: raw.pop(k) for k in list(raw) if k in known}
        return cls(
            name=name,
            prompts=prompts,
            items=list(items),
            easy_count=easy_count,
            generation=gen,
            results_dir=results_dir,
            extra=raw,
            **fields,
        )

    def summary(self) -> str:
        buckets = ["easy" if i < self.easy_count else "hard" for i in range(len(self.items))]
        rows = "\n".join(f"  [{i:2d}] {it:<14} {b:<5}  {p}" for i, (it, b, p) in enumerate(zip(self.items, buckets, self.prompts)))
        return f"config '{self.name}'  model={self.model_id}  fallback={self.fallback_word!r}\n{rows}"
