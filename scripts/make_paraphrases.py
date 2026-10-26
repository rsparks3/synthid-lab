"""Batch-generate the activity materials.

For every story in data/stories.json, produce N watermarked paraphrases and one
unwatermarked control paraphrase, score all of them (plus the human original) with
the detector, and write data/paraphrases.json. Run this once on the DGX Spark before
the session; hand participants the watermarked paraphrases.

    python scripts/make_paraphrases.py --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthid_lab.config import DATA_DIR, load_config  # noqa: E402
from synthid_lab.watermark import Generator           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=Path, default=DATA_DIR / "stories.json")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "paraphrases.json")
    ap.add_argument("--n", type=int, default=3, help="watermarked paraphrases per story")
    ap.add_argument("--controls", type=int, default=1, help="unwatermarked paraphrases per story")
    args = ap.parse_args()

    cfg = load_config()
    gen = Generator(cfg)
    stories = json.loads(args.stories.read_text(encoding="utf-8"))
    results = []
    print(f"{'story':<10} {'kind':<14} {'z':>7} {'mean_g':>7} {'scored':>7}  verdict")
    for s in stories:
        rows = [("original", s["text"])]
        for i in range(args.n):
            rows.append((f"watermarked_{i}", gen.paraphrase(s["text"], watermark=True, seed=1000 + i)))
        for i in range(args.controls):
            rows.append((f"control_{i}", gen.paraphrase(s["text"], watermark=False, seed=2000 + i)))
        for kind, text in rows:
            d = gen.detector.score(text).as_dict()
            print(f"{s['id']:<10} {kind:<14} {d['z']:>7.2f} {d['mean_g']:>7.3f} {d['n_scored']:>7}  {d['verdict']}")
            results.append({"story_id": s["id"], "kind": kind, "text": text,
                            "z": d["z"], "mean_g": d["mean_g"], "n_scored": d["n_scored"], "verdict": d["verdict"]})
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
