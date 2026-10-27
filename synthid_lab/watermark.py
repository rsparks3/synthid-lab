"""Watermarked generation and key-based detection built on transformers' SynthID-Text.

Two independent pieces:

* Generator  - needs the LLM. Paraphrases a story with (or without) the watermark.
* Detector   - needs ONLY the tokenizer + the secret keys. No model forward pass:
               g-values are a pure function of (token n-grams, keys), so detection
               is milliseconds and could run on a laptop.

Scoring uses the "mean g-value" detector from the SynthID paper: under the null
(unwatermarked text) every g-value is an independent fair coin, so the mean is 0.5
and we can report a z-score / one-sided p-value. Watermarked text pushes the mean
above 0.5. Positions whose (ngram_len-1)-token context repeats earlier in the text
are masked out, exactly as the reference implementation does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

from .config import LabConfig, load_prompt



def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_watermark_config(cfg: LabConfig) -> SynthIDTextWatermarkingConfig:
    return SynthIDTextWatermarkingConfig(
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        sampling_table_size=cfg.sampling_table_size,
        sampling_table_seed=cfg.sampling_table_seed,
        context_history_size=cfg.context_history_size,
    )


def make_processor(cfg: LabConfig, device: str | None = None) -> SynthIDTextWatermarkLogitsProcessor:
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=cfg.ngram_len,
        keys=cfg.keys,
        sampling_table_size=cfg.sampling_table_size,
        sampling_table_seed=cfg.sampling_table_seed,
        context_history_size=cfg.context_history_size,
        device=device or _device(),
    )


# --------------------------------------------------------------------------- detection

@dataclass
class DetectionResult:
    n_tokens: int               # tokens in the submitted text
    n_scored: int               # positions that contributed (after context-repetition masking)
    depth: int                  # number of keys / tournament layers
    mean_g: float               # mean g-value; 0.5 = no watermark, higher = watermarked
    z: float                    # z-score vs. the fair-coin null
    p_value: float              # one-sided p-value
    verdict: str                # "watermarked" | "weak" | "not_watermarked" | "too_short"
    tokens: list[str] = field(default_factory=list)                  # decoded tokens, for the UI heatmap
    token_scores: list[float | None] = field(default_factory=list)   # per-token mean g (None = unscored)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def verdict_for(z: float, n_scored: int, min_scored: int = 20) -> str:
    if n_scored < min_scored:
        return "too_short"
    if z >= 4.0:
        return "watermarked"
    if z >= 2.0:
        return "weak"
    return "not_watermarked"


class Detector:
    """Key-based SynthID detector. Only needs the tokenizer that the generator used."""

    def __init__(self, cfg: LabConfig, tokenizer=None, device: str | None = None):
        self.cfg = cfg
        self.device = device or "cpu"   # detection is cheap; CPU is fine everywhere
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_id)
        self.processor = make_processor(cfg, self.device)

    @torch.no_grad()
    def score(self, text: str) -> DetectionResult:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        n_tokens = int(ids.shape[1])
        depth = self.cfg.depth
        ngram = self.cfg.ngram_len
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        tokens = [self.tokenizer.convert_tokens_to_string([t]) for t in raw_tokens]

        if n_tokens < ngram:
            return DetectionResult(n_tokens, 0, depth, 0.5, 0.0, 1.0, "too_short", tokens, [None] * n_tokens)

        # g_values: [1, n_tokens - ngram + 1, depth]  (one row per n-gram ending position)
        g = self.processor.compute_g_values(input_ids=ids)
        # mask: 1 where the context has NOT been seen before in this text, 0 where it repeats
        mask = self.processor.compute_context_repetition_mask(input_ids=ids)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos_mask = self.processor.compute_eos_token_mask(input_ids=ids, eos_token_id=eos_id)
            mask = mask * eos_mask[:, ngram - 1:]

        g = g[0].float()            # [P, depth]
        mask = mask[0].bool()       # [P]
        n_scored = int(mask.sum().item())
        if n_scored == 0:
            return DetectionResult(n_tokens, 0, depth, 0.5, 0.0, 1.0, "too_short", tokens, [None] * n_tokens)

        per_pos = g.mean(dim=1)                      # [P], mean over layers
        mean_g = float(per_pos[mask].mean().item())
        n = n_scored * depth
        z = (mean_g - 0.5) * 2.0 * math.sqrt(n)      # Bernoulli(0.5) null: sd of mean = 0.5/sqrt(n)
        p = 0.5 * math.erfc(z / math.sqrt(2.0))      # one-sided

        token_scores: list[float | None] = [None] * n_tokens
        for i in range(per_pos.shape[0]):
            if mask[i]:
                token_scores[i + ngram - 1] = round(float(per_pos[i].item()), 3)

        return DetectionResult(
            n_tokens=n_tokens, n_scored=n_scored, depth=depth, mean_g=round(mean_g, 4),
            z=round(z, 2), p_value=float(f"{p:.3g}"), verdict=verdict_for(z, n_scored),
            tokens=tokens, token_scores=token_scores,
        )


# --------------------------------------------------------------------------- generation

class Generator:
    """Loads the LLM once; paraphrases with or without the watermark."""

    def __init__(self, cfg: LabConfig, device: str | None = None, dtype: torch.dtype | None = None):
        self.cfg = cfg
        self.device = device or _device()
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype, device_map=self.device)
        self.model.eval()
        self.wm_config = make_watermark_config(cfg)
        self.detector = Detector(cfg, tokenizer=self.tokenizer, device="cpu")

    def build_messages(self, story: str, system: str | None = None) -> list[dict[str, str]]:
        """The exact chat sent to the LLM. The watermark is NOT in here; it is applied at sampling time."""
        return [{"role": "system", "content": system or load_prompt()}, {"role": "user", "content": story}]

    def render_prompt(self, story: str, system: str | None = None) -> str:
        """The raw string the model actually sees, after the chat template is applied."""
        return self.tokenizer.apply_chat_template(self.build_messages(story, system), add_generation_prompt=True, tokenize=False)

    def _prompt_ids(self, story: str, system: str | None = None) -> torch.Tensor:
        messages = self.build_messages(story, system)
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(self.device)

    @torch.no_grad()
    def paraphrase(self, story: str, watermark: bool = True, seed: int | None = None,
                   max_new_tokens: int | None = None) -> str:
        if seed is not None:
            torch.manual_seed(seed)
        input_ids = self._prompt_ids(story)
        kwargs: dict[str, Any] = dict(
            do_sample=True,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if watermark:
            kwargs["watermarking_config"] = self.wm_config
        out = self.model.generate(input_ids, **kwargs)
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
