"""Sanity-check the detector's null distribution and threshold.

Reads data/paraphrases.json (from make_paraphrases.py) and optionally a folder of
extra human-written .txt files, then prints z-score summaries per kind. Under the
null, z should be ~N(0,1); watermarked rows should sit far above 4.

    python scripts/calibrate.py --human-dir data/human_samples
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthid_lab.config import DATA_DIR, load_config  # noqa: E402
from synthid_lab.watermark import Detector            # noqa: E402


def summarize(name: str, zs: list[float]) -> None:
    if not zs:
        return
    mu = st.mean(zs)
    sd = st.pstdev(zs) if len(zs) > 1 else 0.0
    print(f"{name:<16} n={len(zs):<4} mean z={mu:6.2f}  sd={sd:5.2f}  min={min(zs):6.2f}  max={max(zs):6.2f}  "
          f"frac z>=4: {sum(z >= 4 for z in zs) / len(zs):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrases", type=Path, default=DATA_DIR / "paraphrases.json")
    ap.add_argument("--human-dir", type=Path, default=None)
    args = ap.parse_args()

    groups: dict[str, list[float]] = {}
    if args.paraphrases.exists():
        for row in json.loads(args.paraphrases.read_text(encoding="utf-8")):
            kind = row["kind"].rsplit("_", 1)[0]
            groups.setdefault(kind, []).append(row["z"])

    if args.human_dir:
        det = Detector(load_config())
        zs = [det.score(p.read_text(encoding="utf-8")).z for p in sorted(args.human_dir.glob("*.txt"))]
        groups["human_extra"] = zs

    for name, zs in groups.items():
        summarize(name, zs)
    print("\nRule of thumb: z >= 4 means p < 3e-5 under the null. If any 'original', 'control' or "
          "'human_extra' row exceeds 4, something is off (wrong tokenizer? keys?).")


if __name__ == "__main__":
    main()
