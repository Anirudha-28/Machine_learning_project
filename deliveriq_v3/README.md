# DeliverIQ — Delivery Prediction App

## Chat assistant now runs on Gemini (free tier), not Ollama

`llm.py`, `agent.py`, `admin_agent.py`, and `rag.py` were switched from a
local Ollama model to Google's free Gemini API — this is what fixes the
slow chatbot responses:

- **Model:** `gemini-2.5-flash-lite` by default (fastest + highest free-tier
  quota: 15 req/min, 1,000 req/day as of writing). Falls back automatically
  to `gemini-2.5-flash` if flash-lite's quota is hit. Override with
  `export GEMINI_MODEL=gemini-2.5-flash` if you want the slightly stronger
  model and don't mind a bit more latency.
- **Native function calling** replaced the old manual "reply with a JSON
  object" ReAct loop. The old loop's biggest slowdown was retrying an
  entire LLM round trip whenever the small local model wrote invalid JSON
  or chatted instead of complying. Gemini returns a structured tool call
  directly, so that retry cost is gone.
- **Embeddings** (`rag.py`) moved from Ollama's `nomic-embed-text` to
  Gemini's free `text-embedding-004` model — one less local service to run.
  The keyword-search fallback (used if the embedding call fails or
  `vector_store.pkl` hasn't been built) is unchanged.

### Setup — all you need is one free API key

1. Get a free key (no credit card) at **https://aistudio.google.com/apikey**
2. Set it as an environment variable before running the app:
   ```bash
   export GEMINI_API_KEY=your-key-here
   # Windows PowerShell:  $env:GEMINI_API_KEY="your-key-here"
   ```
3. Run the app as usual — `python app.py`. No `ollama serve`, no model pull,
   nothing else to install (still just `pip install -r requirements.txt`).
4. If you rebuild the RAG index (`python rag.py`), it now embeds via Gemini
   too, so make sure `GEMINI_API_KEY` is set for that step as well.

**Free tier limits to know:** flash-lite gives ~15 requests/minute and
~1,000/day per Google account/project. A single chat turn now costs at most
`MAX_STEPS` (3) requests instead of possibly 2x that under the old retry
loop, so you get roughly 300+ real chat turns/day comfortably inside the
free quota. If you ever see a "quota hit" error, it means both flash-lite
and the flash fallback are rate-limited — wait a minute (per-minute limit)
or the rest of the day (daily limit).

## What was wrong before

Your `.pkl` was trained on ~90 engineered columns (log-transforms, target-encoded
category, one-hot dummies for 10 categorical fields, haversine distance, calendar
flags) — but the UI was sending raw form values. Almost every one-hot column
resolved to its dropped reference category and the target-encoding columns were
missing, so the model always landed in the same narrow leaf region → stuck
predicting 2-4 days regardless of input.

## What this app does differently

1. **`feature_engineering.py`** rebuilds the exact schema, confirmed by reading
   the feature names embedded directly in your `.pkl` binary and cross-checking
   every categorical column's unique values against the training CSV.
2. **Operational fields the customer never enters** (traffic, weather, courier
   assignment, warehouse state, road condition, fuel price) are **sampled from
   real historical combinations** in the CSV — 40 scenarios per prediction —
   instead of hard-coded to one default. This is what actually fixes the
   "always 2-4 days" bug: a single fixed default keeps the model in one leaf;
   sampling real historical variation lets it explore the full tree.
3. **`app.py` reorders columns to `model.feature_names_`** at prediction time
   as a safety net, so even if the reconstructed order differs from training,
   the model still gets the right value in the right slot.

## Setup

```bash
pip install -r requirements.txt

# Optional: set up MySQL logging (app works fine without this — logging just
# gets skipped and printed to console instead)
mysql -u root -p < schema.sql
```

Set DB credentials via environment variables if not using defaults:
```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=yourpassword
export DB_NAME=delivery_prediction_db
```

## Test the model BEFORE running the full app

```bash
python test_model.py
```

This prints 10 predictions across different products/states/priorities and
their spread. You should see meaningfully different numbers (not clustered
in a 2-day window). If it's still clustered, the printout tells you which
category the model landed on so you can debug from there.

## Run the app

```bash
python app.py
```

Visit `http://localhost:5000`

## Structure

```
app.py                  Flask routes: home, product, predict, dashboard, chat stub
feature_engineering.py  Rebuilds the model's exact training schema
templates/               HTML pages
static/                  CSS + JS (chat widget)
schema.sql               MySQL table for prediction) logging
test_model.py             Standalone sanity check script
requirements.txt
```

## Still to do

- Wire the `/chat` endpoint to an actual LLM + RAG pipeline (currently a stub
  that echoes the message back)
- Add authentication if this needs to be multi-user
- Consider caching `FeatureBuilder` lookups to disk so app startup is instant
  on restart (currently rebuilt from the CSV every time the app starts)
