"""Configuration for the SynthID lab.

Everything secret (the watermark keys) lives in data/keys.json, which is git-ignored.
Everything else can be overridden with environment variables.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEYS_PATH = Path(os.environ.get("SYNTHID_KEYS_PATH", DATA_DIR / "keys.json"))


@dataclass
class LabConfig:
    # Any HF causal LM with a chat template works. Gemma is thematically nice (SynthID is Google's),
    # but it is gated. Qwen2.5-3B-Instruct is ungated, paraphrases well, and at ~6 GB fits on the
    # DGX Spark next to a resident vLLM server; use the 7B if the memory is free.
    model_id: str = os.environ.get("SYNTHID_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    # Watermark parameters. ngram_len is the big robustness knob:
    #   5 = Google's default (strong secrecy, fragile to edits)
    #   3 = more edit-tolerant (good for a classroom "how much can I change it?" activity)
    # The detector MUST use the same ngram_len and keys as the generator.
    ngram_len: int = int(os.environ.get("SYNTHID_NGRAM_LEN", "3"))
    keys: list[int] = field(default_factory=list)
    sampling_table_size: int = 65536
    sampling_table_seed: int = 0
    context_history_size: int = 1024

    # Sampling. Higher temperature = more entropy = stronger watermark (and wilder prose).
    temperature: float = float(os.environ.get("SYNTHID_TEMPERATURE", "0.9"))
    top_k: int = int(os.environ.get("SYNTHID_TOP_K", "40"))
    max_new_tokens: int = int(os.environ.get("SYNTHID_MAX_NEW_TOKENS", "400"))
    # Length the paraphrase prompt asks for. Detection power grows with sqrt(tokens).
    target_words: int = int(os.environ.get("SYNTHID_TARGET_WORDS", "50"))

    @property
    def depth(self) -> int:
        return len(self.keys)


def load_keys(path: Path = KEYS_PATH) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No watermark keys at {path}. Run:  python scripts/make_keys.py"
        )
    data = json.loads(path.read_text())
    keys = data["keys"] if isinstance(data, dict) else data
    if not keys or not all(isinstance(k, int) for k in keys):
        raise ValueError("keys.json must contain a non-empty list of ints")
    return list(keys)


def load_config() -> LabConfig:
    cfg = LabConfig()
    cfg.keys = load_keys()
    return cfg
