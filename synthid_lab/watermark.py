"""Watermarked generation and key-based detection built on transformers' SynthID-Text.

Two independent pieces:

* Generator  - needs the LLM. Paraphrases a story with (or without) the watermark.
* Detector   - needs ONLY the tokenizer + the secret keys. No model forward pass:
               g-values are a pure function of (token n-grams, keys), so detection
               is milliseconds and could run on a laptop.

Scoring uses the "weighted mean" detector from the SynthID paper: under the null
(unwatermarked text) every g-value is an independent fair coin, so any fixed weighting
of them has mean 0.5 and we can report a z-score / one-sided p-value. Watermarked text
pushes it above 0.5. Weights fall linearly from 10 (first tournament layer) to 1 (last),
because the early layers of the tournament carry most of the signal. Positions whose
(ngram_len-1)-token context repeats earlier in the text are masked out, exactly as the
reference implementation does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

from .config import LabConfig

# Small models write roughly as many words as the input unless told explicitly to expand or condense,
# so the instruction depends on the target. Generator.generate() also enforces the length.
# Detection power grows with sqrt(tokens): short targets make the watermark hard to detect.
PARAPHRASE_SYSTEM = (
    "You are a creative editor. Retell the user's short story in your own words in about {words} "
    "words{how}. Keep the same characters and point of view, but change the wording and sentence "
    "structure throughout. Output only the retold story, no title, no preamble."
)
_EXPAND = ", keeping every plot point and adding sensory detail and inner thoughts to reach the length"
_CONDENSE = ", condensing it to its essential plot points"
_SENTENCE_END = (".", "!", "?", '"', "'", "\u201d", "\u2019")


class _StopAtLength(StoppingCriteria):
    """Stop at the first sentence end once the generated text reaches `words` words."""

    def __init__(self, tokenizer, prompt_len: int, words: int):
        self.tokenizer, self.prompt_len, self.words = tokenizer, prompt_len, words

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True).rstrip()
        # An even number of straight double quotes means we're not stopping inside an open quotation.
        done = len(text.split()) >= self.words and text.endswith(_SENTENCE_END) and text.count('"') % 2 == 0
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


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
    """Build the processor with its sampling table seeded on the CPU, then move it to `device`.

    transformers seeds the g-value sampling table with torch.Generator(device), and the CUDA and
    CPU generators produce different tables for the same seed. Text watermarked on the GPU would
    then be undetectable by a CPU detector. Always seeding on the CPU keeps every device in sync.
    """
    device = device or _device()
    proc = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=cfg.ngram_len,
        keys=cfg.keys,
        sampling_table_size=cfg.sampling_table_size,
        sampling_table_seed=cfg.sampling_table_seed,
        context_history_size=cfg.context_history_size,
        device="cpu",
    )
    if device != "cpu":
        proc.sampling_table = proc.sampling_table.to(device)
        proc.keys = proc.keys.to(device)
        proc.device = device
    return proc


# --------------------------------------------------------------------------- detection

@dataclass
class DetectionResult:
    n_tokens: int               # tokens in the submitted text
    n_scored: int               # positions that contributed (after context-repetition masking)
    depth: int                  # number of keys / tournament layers
    mean_g: float               # plain mean g-value; 0.5 = no watermark, higher = watermarked
    weighted_g: float           # layer-weighted mean g-value (what the z-score is based on)
    z: float                    # z-score of the weighted mean vs. the fair-coin null
    p_value: float              # one-sided p-value
    verdict: str                # "watermarked" | "weak" | "not_watermarked" | "too_short"
    tokens: list[str] = field(default_factory=list)                  # decoded tokens, for the UI heatmap
    token_scores: list[float | None] = field(default_factory=list)   # per-token weighted g (None = unscored)
    token_z: list[float | None] = field(default_factory=list)        # per-token share of z (sums to z)

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

    def with_keys(self, keys: list[int]) -> "Detector":
        """Same tokenizer and settings, different key set (for the wrong-key demonstration)."""
        return Detector(replace(self.cfg, keys=keys), tokenizer=self.tokenizer, device=self.device)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def ngram_survival(self, text: str, reference: str) -> tuple[list[bool | None], list[bool | None]]:
        """For each token, whether the n-gram ending at it also occurs in the other text.

        Returns (survived, kept): `survived` is per token of `text` (does this n-gram come from the
        reference?), `kept` is per token of `reference` (did this n-gram make it into `text`?).
        Only n-grams carry the watermark, so this is exactly what an edit preserved.
        """
        n = self.cfg.ngram_len

        def grams(ids: list[int]) -> list[tuple[int, ...] | None]:
            return [None] * min(n - 1, len(ids)) + [tuple(ids[i - n + 1:i + 1]) for i in range(n - 1, len(ids))]

        a, b = grams(self._ids(text)), grams(self._ids(reference))
        sa, sb = {g for g in a if g}, {g for g in b if g}
        return [None if g is None else g in sb for g in a], [None if g is None else g in sa for g in b]

    @torch.no_grad()
    def score(self, text: str) -> DetectionResult:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        n_tokens = int(ids.shape[1])
        depth = self.cfg.depth
        ngram = self.cfg.ngram_len
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        tokens = [self.tokenizer.convert_tokens_to_string([t]) for t in raw_tokens]

        empty = DetectionResult(n_tokens, 0, depth, 0.5, 0.5, 0.0, 1.0, "too_short", tokens,
                                [None] * n_tokens, [None] * n_tokens)
        if n_tokens < ngram:
            return empty

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
            return empty

        w = torch.linspace(10.0, 1.0, depth)           # layer weights, as in the paper
        per_pos = (g * w).sum(dim=1) / w.sum()          # [P], weighted mean over layers
        mean_g = float(g[mask].mean().item())
        weighted_g = float(per_pos[mask].mean().item())
        # Bernoulli(0.5) null: each position's weighted sum has sd 0.5 * ||w||, independent across positions.
        scale = float(w.sum() / (0.5 * w.norm()))
        contrib = (per_pos - 0.5) * scale / math.sqrt(n_scored)
        z = float(contrib[mask].sum().item())
        p = 0.5 * math.erfc(z / math.sqrt(2.0))         # one-sided

        token_scores: list[float | None] = [None] * n_tokens
        token_z: list[float | None] = [None] * n_tokens
        for i in range(per_pos.shape[0]):
            if mask[i]:
                token_scores[i + ngram - 1] = round(float(per_pos[i].item()), 3)
                token_z[i + ngram - 1] = round(float(contrib[i].item()), 4)

        return DetectionResult(
            n_tokens=n_tokens, n_scored=n_scored, depth=depth, mean_g=round(mean_g, 4),
            weighted_g=round(weighted_g, 4), z=round(z, 2), p_value=float(f"{p:.3g}"),
            verdict=verdict_for(z, n_scored), tokens=tokens, token_scores=token_scores, token_z=token_z,
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

    def _prompt_ids(self, story: str, words: int) -> torch.Tensor:
        n = len(story.split())
        how = _EXPAND if words > 1.2 * n else _CONDENSE if words < 0.8 * n else ""
        messages = [{"role": "system", "content": PARAPHRASE_SYSTEM.format(words=words, how=how)},
                    {"role": "user", "content": story}]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(self.device)

    def _trim(self, new_ids: torch.Tensor) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        out: list[int] = []
        for t in new_ids.tolist():
            if t in eos:
                break
            out.append(t)
        return out

    @torch.no_grad()
    def generate(self, story: str, watermark: bool = True, seed: int | None = None,
                 max_new_tokens: int | None = None, temperature: float | None = None,
                 top_k: int | None = None, words: int | None = None) -> tuple[torch.Tensor, list[int]]:
        """Returns (prompt_ids, generated token ids without the trailing EOS)."""
        if seed is not None:
            torch.manual_seed(seed)
        words = words or self.cfg.target_words
        input_ids = self._prompt_ids(story, words)
        # Hard cap in proportion to the target (English prose is ~1.3 tokens per word), leaving room
        # for the stopping criterion to reach the end of the current sentence.
        cap = min(max_new_tokens or self.cfg.max_new_tokens, int(words * 1.6) + 24)
        kwargs: dict[str, Any] = dict(
            do_sample=True,
            temperature=temperature or self.cfg.temperature,
            top_k=top_k or self.cfg.top_k,
            # Disable the model's own top_p / repetition penalty so temperature and top_k are the
            # only knobs, and so tournament() can reconstruct each step exactly.
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=cap,
            # Small models often wrap up early; don't allow EOS until roughly 85% of the target.
            min_new_tokens=int(words * 1.1),
            stopping_criteria=StoppingCriteriaList([_StopAtLength(self.tokenizer, input_ids.shape[1], words)]),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if watermark:
            # A custom processor of the same type replaces the one generate() would build from
            # watermarking_config, in the same slot (after top_k / temperature). Fresh per call
            # because the processor keeps per-sequence state.
            kwargs["watermarking_config"] = self.wm_config
            kwargs["logits_processor"] = LogitsProcessorList([make_processor(self.cfg, self.device)])
        out = self.model.generate(input_ids, **kwargs)
        ids = self._trim(out[0, input_ids.shape[1]:])
        if len(ids) >= cap:
            ids = self._to_sentence_end(ids)
        return input_ids, ids

    def _to_sentence_end(self, ids: list[int]) -> list[int]:
        """Cut a capped generation back to its last complete sentence, if that keeps at least half."""
        for i in range(len(ids), len(ids) // 2, -1):
            if self.tokenizer.decode(ids[:i], skip_special_tokens=True).rstrip().endswith(_SENTENCE_END):
                return ids[:i]
        return ids

    def paraphrase(self, story: str, watermark: bool = True, seed: int | None = None,
                   max_new_tokens: int | None = None) -> str:
        _, ids = self.generate(story, watermark=watermark, seed=seed, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def tournament(self, prompt_ids: torch.Tensor, gen_ids: list[int], temperature: float,
                   top_k: int, n_show: int = 8) -> list[dict[str, Any]]:
        """Reconstruct, for every generated token, the candidates and how the watermark reweighted them.

        One forward pass over prompt + output gives the model's distribution at each step; we apply
        temperature and top_k exactly as generate() did, compute every candidate's g-values from the
        same context the watermark processor saw (the previous ngram_len-1 *generated* tokens,
        zero-padded at the start), and run the processor's own tournament update on them.
        """
        if not gen_ids:
            return []
        n, depth = self.cfg.ngram_len, self.cfg.depth
        P, T = prompt_ids.shape[1], len(gen_ids)
        full = torch.cat([prompt_ids, torch.tensor([gen_ids], device=self.device)], dim=1)
        logits = self.model(full).logits[0, P - 1:P - 1 + T].float() / temperature   # [T, V]
        top_vals, top_idx = logits.topk(top_k, dim=-1)                                  # [T, k]

        proc = make_processor(self.cfg, self.device)
        padded = torch.tensor([0] * (n - 1) + gen_ids, device=self.device)
        ctx = padded.unfold(0, n - 1, 1)[:T]                                            # [T, n-1]
        ngrams = torch.cat([ctx[:, None, :].expand(T, top_k, n - 1), top_idx[:, :, None]], dim=2)
        g = proc.sample_g_values(proc.compute_ngram_keys(ngrams.reshape(1, T * top_k, n)))
        g = g.reshape(T, top_k, depth).float()                                          # [T, k, depth]
        p = torch.softmax(top_vals, dim=-1)
        p_wm = torch.exp(proc.update_scores(top_vals, g))

        steps: list[dict[str, Any]] = []
        seen: set[tuple[int, ...]] = set()
        for t in range(T):
            key = tuple(ctx[t].tolist())
            repeated = key in seen   # the processor skips watermarking on a repeated context
            seen.add(key)
            order = list(range(min(n_show, top_k)))
            chosen = (top_idx[t] == gen_ids[t]).nonzero()
            if chosen.numel() and int(chosen[0]) not in order:
                order.append(int(chosen[0]))
            steps.append({
                "token": self.tokenizer.decode([gen_ids[t]]),
                "repeated_context": repeated,
                "candidates": [{
                    "token": self.tokenizer.decode([int(top_idx[t, j])]),
                    "chosen": int(top_idx[t, j]) == gen_ids[t],
                    "p": round(float(p[t, j]), 4),
                    "p_wm": round(float(p[t, j] if repeated else p_wm[t, j]), 4),
                    "g_mean": round(float(g[t, j].mean()), 3),
                    "g_bits": "".join(str(int(b)) for b in g[t, j].tolist()),
                } for j in order],
            })
        return steps
