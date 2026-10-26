# SynthID-Text classroom lab

A self-hosted SynthID-Text setup for a "can the detector still see the watermark after *you* rewrite it?" activity.

* **Generator** (DGX Spark): an open LLM paraphrases your short stories with Google's SynthID-Text
  watermark, using a secret key set that you control.
* **Detector** (anywhere): a web page where participants paste text. The server scores it against the
  secret key and shows a verdict, a z-score, and a per-token heatmap of watermark evidence.

Everything is built on the production SynthID-Text implementation inside Hugging Face `transformers`
(`SynthIDTextWatermarkingConfig` / `SynthIDTextWatermarkLogitsProcessor`). No training is needed: the
detector is the paper's "mean g-value" test, which only needs the tokenizer and the keys.

## How it works in one paragraph

SynthID-Text changes *which token the model samples* at each step. For every position it hashes the
previous `ngram_len - 1` tokens together with each secret key to produce a set of pseudo-random bits
("g-values"), then runs a tournament that prefers candidate tokens whose g-values are 1. Unwatermarked
text has g-values that are fair coins (mean 0.5). Watermarked text has a mean noticeably above 0.5. The
detector re-derives the g-values from the pasted tokens and the keys and reports how far above 0.5 the
mean is as a z-score. z >= 4 means p < 0.00003 under the "no watermark" hypothesis.

Consequence for the activity: the watermark lives in exact token n-grams. Pasting the LLM output
verbatim lights the detector up. Swapping a few words weakens it. Rewriting the story in your own words
destroys it, because almost none of the original n-grams survive. That is the point worth teaching,
so plan the activity as a gradient (see "Running the activity").

## Layout

```
synthid_lab/config.py      model id, ngram_len, sampling settings (env-overridable); loads data/keys.json
synthid_lab/watermark.py   Generator (needs the LLM) and Detector (tokenizer + keys only)
synthid_lab/server.py      FastAPI: GET /  POST /api/detect  POST /api/generate  GET /api/info
static/index.html          participant page (+ collapsible instructor panel)
scripts/make_keys.py       create the secret key set (data/keys.json, git-ignored)
scripts/make_paraphrases.py batch-generate watermarked + control paraphrases for every story
scripts/calibrate.py       check the null distribution / threshold
scripts/smoke_test.py      end-to-end check with a 360M model on CPU
data/stories.json          your ~50-word stories (3 placeholders included)
Dockerfile                 DGX Spark image based on the NGC PyTorch container
```

## Setup on the DGX Spark

The Spark is aarch64 + Blackwell (sm_121) with CUDA 13, so use NVIDIA's container rather than a
PyPI torch wheel. Do **not** `pip install torch` inside it; that replaces NVIDIA's build with a CPU one.

```bash
git clone <this repo> synthid-lab && cd synthid-lab
python3 scripts/make_keys.py                       # writes data/keys.json  (keep private!)
docker build -t synthid-lab .
docker run --gpus all -p 8000:8000 \
  -v $PWD/data:/app/data \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e SYNTHID_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e SYNTHID_ADMIN_TOKEN=change-me \
  synthid-lab
```

Then open `http://<spark-ip>:8000`. The detector works immediately; the model downloads and loads on
the first paraphrase request (about 15 GB for the 7B model, a minute or two to load).

Alternative without Docker (uv, cu130 aarch64 wheels):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
python scripts/make_keys.py
SYNTHID_ADMIN_TOKEN=change-me uvicorn synthid_lab.server:app --host 0.0.0.0 --port 8000
```

Model choice: anything with a chat template works. `Qwen/Qwen2.5-7B-Instruct` is ungated and a good
paraphraser. `google/gemma-3-12b-it` is thematically nice (SynthID is Google's) but gated: accept the
license on Hugging Face and set `HF_TOKEN`. The Spark's 128 GB fits either in bf16 with room to spare.

### Batch-generate the materials

```bash
# inside the container or venv, on the Spark
python scripts/make_paraphrases.py --n 3 --controls 1
python scripts/calibrate.py
```

`make_paraphrases.py` writes `data/paraphrases.json` with, per story: the human original, three
watermarked paraphrases, and one unwatermarked control, each with its detector score. Print the
watermarked ones for participants. `calibrate.py` prints z-score summaries per kind; originals and
controls should hover around 0, watermarked rows should be well above 4.

## Local development (Windows / laptop)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\make_keys.py
.venv\Scripts\python scripts\smoke_test.py          # ~1 min on CPU, downloads SmolLM2-360M
```

Run the detector page locally against the same keys and tokenizer (no GPU needed):

```powershell
$env:SYNTHID_MODE="detect"; .venv\Scripts\python -m uvicorn synthid_lab.server:app --port 8000
```

## Knobs that matter

| Setting | Where | Effect |
|---|---|---|
| `SYNTHID_NGRAM_LEN` | env / `config.py` | Context length hashed per token. 5 is Google's default and is fragile to edits. 3 survives more editing and is better for a classroom gradient. Generator and detector must match. |
| keys (`data/keys.json`) | `make_keys.py` | Length = watermark depth (30 by default, as in the paper). More layers = stronger signal per token. |
| `SYNTHID_TEMPERATURE` | env | Higher = more entropy = stronger watermark. 0.9 to 1.0 works well. |
| paraphrase length | `PARAPHRASE_SYSTEM` prompt | Detection power grows with sqrt(tokens). 150 to 200 words gives comfortable margins; a 50-word output is borderline. |
| thresholds | `verdict_for()` | z >= 4 "watermarked", 2 to 4 "weak", else "no watermark". |

## Running the activity

Suggested flow, which turns the inevitable "no watermark" on a full rewrite into the lesson:

1. **Verbatim.** Participants paste the watermarked paraphrase as given. Detector: strong (z of 8 to 15 for 150 to 200 words).
2. **Light edit.** Change five to ten words. Detector: usually still "watermarked" or "weak". Ask them to find which tokens kept the heat.
3. **Heavy edit.** Reorder sentences, replace half the words. Detector: "weak" or gone.
4. **Own words.** Rewrite from memory. Detector: "no watermark". Discuss why (the watermark is in token n-grams, not in meaning), and what that implies about watermarks as a defense against determined paraphrasing versus casual copy-paste.
5. **Control.** Paste an unwatermarked LLM paraphrase and the human original: both read as "no watermark", which shows the detector is not an "AI detector" but a *key* detector.

Operational notes:

* The detector endpoint is cheap (tokenize + hash); dozens of simultaneous participants are fine on the Spark or even a laptop.
* Keep `data/keys.json` off participant machines. Anyone with the keys can forge the watermark.
* `SYNTHID_ADMIN_TOKEN` gates `/api/generate` so participants cannot generate fresh watermarked text during the session.
