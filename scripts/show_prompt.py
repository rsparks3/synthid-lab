"""Print the exact prompt the paraphrasing LLM receives (chat template applied), without loading model weights.

    python scripts/show_prompt.py                 # first story in data/stories.json
    python scripts/show_prompt.py --story robot --words 120
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer  # noqa: E402
from synthid_lab.config import DATA_DIR, LabConfig  # noqa: E402
from synthid_lab.watermark import system_message  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--story", default=None, help="story id from data/stories.json")
ap.add_argument("--model", default=LabConfig().model_id)
ap.add_argument("--words", type=int, default=LabConfig().target_words, help="target length (fills {words} / {how})")
args = ap.parse_args()

stories = json.loads((DATA_DIR / "stories.json").read_text(encoding="utf-8"))
story = next((s for s in stories if s["id"] == args.story), stories[0])
tok = AutoTokenizer.from_pretrained(args.model)
messages = [{"role": "system", "content": system_message(story["text"], args.words)}, {"role": "user", "content": story["text"]}]
rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f"# model: {args.model}\n# prompt tokens: {len(tok(rendered, add_special_tokens=False)['input_ids'])}\n")
print(rendered)
