"""FastAPI service: key-based SynthID detector (public) + watermarked paraphraser (instructor).

Run:  uvicorn synthid_lab.server:app --host 0.0.0.0 --port 8000

The detector is loaded at startup (tokenizer only, instant). The LLM is loaded lazily on
the first /api/generate call so the detector page is usable immediately, and so the same
code runs on a laptop in detector-only mode (set SYNTHID_MODE=detect to refuse generation).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, load_config, load_prompt
from .watermark import Detector, Generator

STATIC_DIR = ROOT / "static"
MODE = os.environ.get("SYNTHID_MODE", "full")            # "full" | "detect"
ADMIN_TOKEN = os.environ.get("SYNTHID_ADMIN_TOKEN", "")   # if set, /api/generate requires X-Admin-Token

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


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class GenerateRequest(BaseModel):
    story: str = Field(..., min_length=1, max_length=5000)
    watermark: bool = True
    seed: int | None = None
    max_new_tokens: int | None = Field(default=None, ge=16, le=1024)


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
    }


@app.post("/api/detect")
def detect(req: DetectRequest) -> dict:
    return detector.score(req.text).as_dict()


@app.post("/api/generate")
def generate(req: GenerateRequest, x_admin_token: str = Header(default="")) -> dict:
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Bad or missing X-Admin-Token header.")
    gen = get_generator()
    with _gen_lock:   # one generation at a time; plenty for a classroom
        text = gen.paraphrase(req.story, watermark=req.watermark, seed=req.seed,
                              max_new_tokens=req.max_new_tokens)
    return {"text": text, "watermark": req.watermark, "detection": detector.score(text).as_dict()}
