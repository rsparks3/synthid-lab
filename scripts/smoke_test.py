"""End-to-end check with a tiny model (runs on CPU in a couple of minutes).

Generates one watermarked and one unwatermarked paraphrase, scores both plus the
human original, and fails if the watermarked text is not clearly separated.

    python scripts/smoke_test.py                       # SmolLM2-360M-Instruct
    python scripts/smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthid_lab.config import load_config    # noqa: E402
from synthid_lab.watermark import Generator   # noqa: E402

STORY = (
    "Mara found the lighthouse keeper's logbook in a tide pool, pages swollen with salt. "
    "The last entry, dated the night of the wreck, read only: 'The light was on. They did not turn.' "
    "She climbed the stairs that evening, lit the lamp, and waited to see who would come."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config()
    cfg.model_id = args.model
    gen = Generator(cfg)
    print(f"model={cfg.model_id} device={gen.device} ngram_len={cfg.ngram_len} depth={cfg.depth}\n")

    rows = [("original (human)", STORY)]
    for name, wm in (("watermarked", True), ("unwatermarked", False)):
        t0 = time.time()
        text = gen.paraphrase(STORY, watermark=wm, seed=7, max_new_tokens=args.max_new_tokens)
        print(f"--- {name} ({time.time() - t0:.0f}s) ---\n{text}\n")
        rows.append((name, text))

    scores = {}
    print(f"{'kind':<18} {'tokens':>6} {'scored':>6} {'mean_g':>7} {'z':>7}  p         verdict")
    for name, text in rows:
        d = gen.detector.score(text)
        scores[name] = d.z
        print(f"{name:<18} {d.n_tokens:>6} {d.n_scored:>6} {d.mean_g:>7.3f} {d.z:>7.2f}  {d.p_value:<9.2g} {d.verdict}")

    ok = scores["watermarked"] >= 4.0 and scores["unwatermarked"] < 3.0 and scores["original (human)"] < 3.0
    print("\nPASS" if ok else "\nFAIL: watermarked text not cleanly separated (try more tokens or higher temperature)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
