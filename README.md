# SynthID-Text classroom lab

A self-hosted SynthID-Text setup for a "can the detector still see the watermark after *you* rewrite it?" activity.

* **Generator** (DGX Spark): an open LLM paraphrases your short stories with Google's SynthID-Text
  watermark, using a secret key set that you control.
* **Detector** (anywhere): a web page where participants paste text. The server scores it against the
  secret key and shows a verdict, a z-score, and a per-token heatmap of watermark evidence.

Everything is built on the production SynthID-Text implementation inside Hugging Face `transformers`
(`SynthIDTextWatermarkingConfig` / `SynthIDTextWatermarkLogitsProcessor`). No training is needed: the
detector is the paper's "weighted mean" g-value test, which only needs the tokenizer and the keys.

## What the dashboard shows

The page opens on the **Detector**. Paste any text and it is scored as you type against the lab's
secret key. You get a verdict, the z-score placed on the bell curve of what unwatermarked text scores,
a p-value, and a per-token heatmap: red tokens are n-grams whose key-derived coin flips came up in the
watermark's favor, blue ones went against it. Other tabs show the evidence accumulating across the
text, the most influential words, and an n-gram comparison with the original story.

![Detector view: a watermarked paraphrase scores z = 8.99 and lights up the heatmap](docs/screenshots/detector.png)

The **Key** box is the wrong-key test. Click "Random key" and the very same text drops back to
chance, which is the point: this is a *key* detector, not an "AI detector".

![The same text scored with a random key reads as no watermark](docs/screenshots/detector-wrong-key.png)

"Show the generator & key" opens the **Generator** (instructor side). Pick or write a ~50-word story,
set temperature, top-k, target length and seed, and paraphrase it with or without the watermark. The
output goes straight into the detector. Click any output token to see the tournament that chose it:
the model's own probabilities, how the watermark reweighted them, and the g-value bits per key layer.

![Generator with the per-token tournament view](docs/screenshots/generator.png)

The **How it works** tab is a self-contained explainer for participants, from tokens and n-grams
through the tournament to the z-score, with a suggested lesson flow and references.

![How it works page](docs/screenshots/how-it-works.png)

## Run it locally

Detector only (no GPU, a few hundred MB for the tokenizer):

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/make_keys.py                            # writes data/keys.json (keep it private)
SYNTHID_MODE=detect uvicorn synthid_lab.server:app --port 8000
```

Open <http://localhost:8000>. The detector and the "How it works" page work immediately; the
generator is disabled in this mode. To check the whole pipeline on a CPU, `python scripts/smoke_test.py`
generates and detects with a 360M model in about a minute.

Detector **and** generator (a GPU with ~6 GB free for the default Qwen2.5-3B-Instruct; the model is
downloaded from Hugging Face on first use):

```bash
pip install torch            # or the CUDA wheel for your platform; on a DGX Spark see below
SYNTHID_PRELOAD=1 uvicorn synthid_lab.server:app --host 0.0.0.0 --port 8000
```

Set `SYNTHID_ADMIN_TOKEN=<secret>` to require an `X-Admin-Token` header on `/api/generate` so
participants cannot generate fresh watermarked text during the session. The detector and the generator
must share one `data/keys.json` and one `SYNTHID_NGRAM_LEN`; copy the keys file to any machine that
runs a detector.

Screenshots in `docs/screenshots/` were taken from a live instance; regenerate them with Playwright
against a running server if the UI changes.

## How it works in one paragraph

SynthID-Text changes *which token the model samples* at each step. For every position it hashes the
previous `ngram_len - 1` tokens together with each secret key to produce a set of pseudo-random bits
("g-values"), then runs a tournament that prefers candidate tokens whose g-values are 1. Unwatermarked
text has g-values that are fair coins (mean 0.5). Watermarked text has a mean noticeably above 0.5. The
detector re-derives the g-values from the pasted tokens and the keys and reports how far above 0.5 the
mean is as a z-score, weighting early tournament layers (which carry most of the signal) 10x more than
the last. z >= 4 means p < 0.00003 under the "no watermark" hypothesis.

Consequence for the activity: the watermark lives in exact token n-grams. Pasting the LLM output
verbatim lights the detector up. Swapping a few words weakens it. Rewriting the story in your own words
destroys it, because almost none of the original n-grams survive. That is the point worth teaching,
so plan the activity as a gradient (see "Running the activity").

## Layout

```
synthid_lab/config.py      model id, ngram_len, sampling settings (env-overridable); loads data/keys.json
synthid_lab/watermark.py   Generator (needs the LLM) and Detector (tokenizer + keys only)
synthid_lab/server.py      FastAPI: GET /  POST /api/detect  POST /api/generate  GET /api/info
static/index.html          dashboard: detector (heatmap, z-curve, top words, compare) + "show the generator & key"
scripts/make_keys.py       create the secret key set (data/keys.json, git-ignored)
scripts/make_paraphrases.py batch-generate watermarked + control paraphrases for every story
scripts/calibrate.py       check the null distribution / threshold
scripts/show_prompt.py     print the exact prompt the model receives (no weights loaded)
scripts/smoke_test.py      end-to-end check with a 360M model on CPU
data/prompt.txt            system prompt template for the paraphraser ({words}, {how} placeholders)
data/stories.json          your ~50-word stories (3 placeholders included; shown as examples in the dashboard)
deploy/synthid-lab.service user systemd unit (port 8090, preloads the model)
docs/screenshots/          dashboard screenshots used in this README
Dockerfile                 DGX Spark image based on the NGC PyTorch container
```

## Setup on the DGX Spark (current deployment)

This Spark also runs a vLLM server on port 8000 that holds most of the unified memory, so the lab runs
Qwen2.5-3B-Instruct (~6 GB) in-process next to it, on port **8090**, as a user systemd service:

```bash
uv venv .venv -p 3.12
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -p .venv -r requirements.txt
.venv/bin/python scripts/make_keys.py
cp deploy/synthid-lab.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now synthid-lab
loginctl enable-linger $USER            # start at boot without a login session
journalctl --user -u synthid-lab -f     # logs
```

Open `http://<spark-ip>:8090`. The page opens on the detector only. It re-scores as you type (toggle
"Update as I type"), and its Key box scores against any key you paste or a random one, which is the
wrong-key test; empty means the lab's secret key. Tabs show a word heatmap, the z-score building up
across the text, the most influential words, and a comparison with the original story.

"Show the generator & key" reveals the paraphraser (watermark toggle, temperature / top-k / length /
seed), a per-token tournament view (click a token, then use the arrow keys), and the secret key. Each
paraphrase is copied into the detector automatically.

A hosted OpenAI-compatible endpoint can't be used for the watermarked text: the watermark is applied
inside sampling, which the endpoint does itself.

## Setup on the DGX Spark (Docker, standalone)

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

Model choice (the default is Qwen2.5-3B-Instruct; the 7B needs ~15 GB free): anything with a chat template works. `Qwen/Qwen2.5-7B-Instruct` is ungated and a good
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
| paraphrase length | length slider / `{words}` in `data/prompt.txt` | Detection power grows with sqrt(tokens). 150 words gives z of about 13 to 17; 50 words about 4 to 6; under ~30 words is usually inconclusive. |
| thresholds | `verdict_for()` | z >= 4 "watermarked", 2 to 4 "weak", else "no watermark". |
| system prompt | `data/prompt.txt` (or `SYNTHID_PROMPT_PATH`), editable per request in the dashboard | Template for the paraphrase instructions. `{words}` becomes the target length and `{how}` an expand / condense phrase. `python scripts/show_prompt.py --words 50` prints exactly what the model receives. |
| `SYNTHID_TARGET_WORDS` | env / dashboard slider | Length the paraphrase prompt asks for (default 50, adjustable 10 to 150 in the dashboard). Small models write about as many words as the input unless asked to expand. |

Note on devices: transformers seeds the g-value sampling table with `torch.Generator(device)`, and the
CUDA and CPU generators give different tables for the same seed, so text watermarked on a GPU was
undetectable by a CPU detector. `make_processor()` now always seeds on the CPU and moves the table, and
the generator passes that processor to `generate()`, so a laptop detector matches a GPU generator.

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
