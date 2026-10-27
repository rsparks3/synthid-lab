"""FastAPI service: key-based SynthID detector (public) + watermarked paraphraser (instructor).

Run:  uvicorn synthid_lab.server:app --host 0.0.0.0 --port 8000

The detector is loaded at startup (tokenizer only, instant). The LLM is loaded lazily on
the first /api/generate call so the detector page is usable immediately, and so the same
code runs on a laptop in detector-only mode (set SYNTHID_MODE=detect to refuse generation).
Set SYNTHID_PRELOAD=1 to load the LLM in the background at startup instead.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DATA_DIR, ROOT, load_config, load_prompt
from .watermark import Detector, Generator, system_message

STATIC_DIR = ROOT / "static"
MODE = os.environ.get("SYNTHID_MODE", "full")            # "full" | "detect"
ADMIN_TOKEN = os.environ.get("SYNTHID_ADMIN_TOKEN", "")   # if set, /api/generate requires X-Admin-Token
PRELOAD = os.environ.get("SYNTHID_PRELOAD", "") == "1"

cfg = load_config()
detector = Detector(cfg, device="cpu")
_generator: Generator | None = None
_gen_lock = threading.Lock()

app = FastAPI(title="SynthID Lab", version="0.1")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_generator() -> Generator:
    global _generator
    if MODE != "full":
        raise HTTPException(status_code=403, detail="This server runs in detect-only mode.")
    with _gen_lock:
        if _generator is None:
            _generator = Generator(cfg)
            # Share the tokenizer so detection and generation can never drift apart.
            detector.tokenizer = _generator.tokenizer
    return _generator


if PRELOAD and MODE == "full":
    threading.Thread(target=get_generator, daemon=True).start()


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    reference: str | None = Field(default=None, max_length=20000)   # e.g. the original paraphrase
    # Score with this key set instead of the lab's secret one (e.g. a random key, to show it finds nothing).
    keys: list[Annotated[int, Field(ge=0, lt=2**53)]] | None = Field(default=None, min_length=1, max_length=256)


class GenerateRequest(BaseModel):
    story: str = Field(..., min_length=1, max_length=5000)
    watermark: bool = True
    seed: int | None = None
    max_new_tokens: int | None = Field(default=None, ge=16, le=1024)
    temperature: float | None = Field(default=None, ge=0.1, le=2.0)
    top_k: int | None = Field(default=None, ge=2, le=200)
    words: int | None = Field(default=None, ge=10, le=150)
    system: str | None = Field(default=None, max_length=4000)   # overrides data/prompt.txt for this call


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info() -> dict:
    return {
        "model_id": cfg.model_id,
        "ngram_len": cfg.ngram_len,
        "depth": cfg.depth,
        "mode": MODE,
        "system_prompt": load_prompt(),
        "generator_loaded": _generator is not None,
        "generator_loading": _gen_lock.locked() and _generator is None,
        "defaults": {"temperature": cfg.temperature, "top_k": cfg.top_k, "max_new_tokens": cfg.max_new_tokens,
                     "words": cfg.target_words},
    }


@app.get("/api/stories")
def stories() -> list:
    path = DATA_DIR / "stories.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.get("/api/keys")
def keys() -> dict:
    # Normally secret: anyone holding these can detect and forge the watermark. Shown here on purpose.
    return {
        "keys": cfg.keys,
        "ngram_len": cfg.ngram_len,
        "depth": cfg.depth,
        "sampling_table_size": cfg.sampling_table_size,
        "sampling_table_seed": cfg.sampling_table_seed,
        "context_history_size": cfg.context_history_size,
    }


@app.post("/api/detect")
def detect(req: DetectRequest) -> dict:
    det = detector.with_keys(req.keys) if req.keys else detector
    out = det.score(req.text).as_dict()
    out["secret_key"] = det.cfg.keys == cfg.keys
    if req.reference and req.reference.strip():
        out["survived"], kept = det.ngram_survival(req.text, req.reference)
        out["reference"] = {**det.score(req.reference).as_dict(), "kept": kept}
    return out


@app.post("/api/generate")
def generate(req: GenerateRequest, x_admin_token: str = Header(default="")) -> dict:
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing X-Admin-Token header.")
    gen = get_generator()
    temperature = req.temperature or cfg.temperature
    top_k = req.top_k or cfg.top_k
    words = req.words or cfg.target_words
    system = req.system.strip() if req.system and req.system.strip() else None
    with _gen_lock:   # one generation at a time; plenty for a classroom
        prompt_ids, ids = gen.generate(req.story, watermark=req.watermark, seed=req.seed,
                                       max_new_tokens=req.max_new_tokens, temperature=temperature, top_k=top_k,
                                       words=words, system=system)
        steps = gen.tournament(prompt_ids, ids, temperature=temperature, top_k=top_k)
    text = gen.tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {
        "text": text, "watermark": req.watermark, "detection": detector.score(text).as_dict(),
        "system": system_message(req.story, words, system),   # the exact system message the model got
        "steps": steps, "params": {"temperature": temperature, "top_k": top_k, "seed": req.seed, "words": words},
    }
