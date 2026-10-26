"""Generate a fresh secret SynthID key set into data/keys.json.

The key set is a list of integers; its length is the watermark "depth"
(number of tournament layers). Google's paper uses 30.
Keep this file private: anyone who has it can detect (and forge) your watermark.
"""
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthid_lab.config import KEYS_PATH  # noqa: E402

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 30

if KEYS_PATH.exists():
    print(f"{KEYS_PATH} already exists; delete it first if you really want new keys.")
    sys.exit(1)

keys = [secrets.randbelow(2**31 - 1) for _ in range(DEPTH)]
KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
KEYS_PATH.write_text(json.dumps({"keys": keys}, indent=2))
print(f"Wrote {DEPTH} keys to {KEYS_PATH}")
