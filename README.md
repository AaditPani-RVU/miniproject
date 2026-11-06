# Kisan Sahayak: AI Farmer Advisory Assistant

Indian-language crop, irrigation and fertilizer guidance, grounded in curated agricultural documents (RAG).

```
frontend (static HTML/JS)  ->  FastAPI  ->  [IndicTrans2 in]  ->  multilingual-e5 retrieval
                                         ->  AgriParam (GGUF, llama.cpp/Metal)  ->  [IndicTrans2 out]
```

- **LLM**: BharatGen AgriParam (Hindi + English, 2048-token context) as a 4-bit GGUF, about 1.7 GB.
- **Retrieval**: `intfloat/multilingual-e5-small`, cosine search over ~110-word chunks.
- **Other languages** (Odia, Kannada, Tamil, ...): optional AI4Bharat IndicTrans2 translate-in / translate-out.
- **Safety**: if the best chunk scores below `MIN_SCORE` the bot refuses instead of guessing.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # use .venv/bin/python if `python` is aliased elsewhere
pip install -r requirements.txt

# 1) LLM server: AgriParam Q4_K_M (~1.7 GB) on Metal via llama.cpp
brew install llama.cpp
hf download arunmcops/AgriParam-GGUF AgriParam-Q4_K_M.gguf     # or let huggingface_hub fetch it
llama-server -m <path-to>/AgriParam-Q4_K_M.gguf -c 2048 -ngl 99 --port 8081 --alias agriparam

# 2) API + frontend
cat > .env <<EOF2
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:8081/v1
OPENAI_MODEL=agriparam
EOF2
uvicorn backend.app.main:app --reload     # open http://127.0.0.1:8000
```

`LLM_BACKEND=extractive` runs without any LLM (returns the best passage) for UI/retrieval testing.
`LLM_BACKEND=gguf` loads the model in-process with `llama-cpp-python`, but its Metal wheel and source build
both failed on this machine (corrupt wheel / missing C++ headers), so `llama-server` is the tested path.

Hindi and the other Indian languages use IndicTrans2 (`TRANSLATOR=indictrans2`): question -> English -> retrieve + AgriParam -> answer -> back.
Setup notes:
- The IndicTrans2 repos are **gated** on Hugging Face. Log in (`hf auth login`) and accept the terms on both
  `ai4bharat/indictrans2-indic-en-dist-200M` and `ai4bharat/indictrans2-en-indic-dist-200M`.
- `pip install git+https://github.com/VarunGumma/IndicTransToolkit.git` (not on the main package index).
- Needs `transformers<5`; generation runs with `use_cache=False` because IndicTrans2's remote code breaks on the newer cache classes.

Gotcha: AgriParam's chat template **drops the system role**, so all instructions must live in the single user turn
(see `build_messages` in `backend/app/llm.py`).

## Add real documents
Put PDFs/TXT into `data/docs/` and restart (or `POST /api/reindex`). `iiwbr_cleaned_tables.md` is a hand-cleaned restatement of the IIWBR report's tables (the raw PDF extraction scrambles them); `.md` files are chunked per paragraph, other files per ~110 words.

## Evaluate
Write 30-50 cases in `eval/testcases.jsonl` (mix of answerable, unanswerable, Hindi and English), then:

```bash
python eval/run_eval.py
```

Tune `MIN_SCORE` from the score gap between answerable and unanswerable cases in `results.csv`.

## How answers are produced (source-first)

1. Normalise the input (empty/junk/greeting -> help message; Devanagari or Roman-script Hindi detected; Roman Hindi rewritten to English with a small glossary).
2. Retrieve 8 candidate passages with multilingual-e5, then rerank them with a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
3. Gates: refuse if the reranker score is low, the question names another crop or asks for live/commercial data (prices, weather, schemes), or a key word of the question appears nowhere in the corpus while the reranker is not confident.
4. The answer is the **exact text of the best passage** with its citation. On our test sets that passage held the right fact 95% of the time versus 76% for the model's own wording, and no guard setting made the model's text more accurate than the quote.
5. The model's plain-language summary is shown below, labelled, only if it stays strictly inside the best passage (numbers, names and vocabulary all checked). Hindi shows the translated source text only.

## Results

Two test sets: `eval/heldout.jsonl` (68 cases, written **before** the guards and reranker were built, so it is the honest number) and `eval/testcases.jsonl` (66 cases, used while tuning, so it is optimistic).

| Metric | Held-out (68) | Tuned set (66) |
|---|---|---|
| Retrieval hit@4 (answerable) | 33/33 = 100% | 43/44 = 98% |
| Source passage has the expected fact | 29/33 = 88% | 44/44 = 100% |
| Unanswerable questions answered anyway | 0/35 | 0/22 |
| Answerable questions refused | 3/33 | 0/44 |
| Hindi / Roman-Hindi answerable | 8/8 | 11/11 |

Held-out progression: first version 17/33 (52%) -> reranker + calibrated gates 25/33 (76%) -> source-first answers 28/33 (85%) -> residue and live-data guard fixes 29/33 (88%).

The held-out set deliberately includes hard cases: other crops, wheat topics the corpus lacks (nematodes, export policy, blast), prompt-injection attempts ("ignore your instructions"), greetings, junk input, and Roman-script Hindi.

Remaining held-out failures (4 of 33)
- Wrongly refused: "only one irrigation, when?" (the Table 5 passage is phrased differently), "how much FYM gave the highest yield" and Hindi "urea kab dalni chahiye".
- Related but not the best passage: cost of production per quintal and "does zero tillage save money" return a neighbouring passage.

Caveats for the report
- Keyword match is a proxy for correctness. Read `eval/*.csv` by hand.
- Vocabulary and crop lists are hand-written and can still misfire on unusual wording; a false refusal is safe but annoying.
- The model's own wording was wrong often enough (24% on our sets, e.g. a storage answer that said 14% moisture and "sprinkle water" where the bulletin says below 12%) that it is not used as the answer.
- Files: `eval/heldout_baseline.csv` (first version), `eval/heldout_v4.csv` (final), `eval/results_final.csv` (tuned set, final).

## Layout
```
backend/app/   config, retrieval, llm, translate, rag, main (FastAPI)
frontend/      index.html, style.css, app.js
data/docs/     source documents
eval/          testcases.jsonl, run_eval.py
```
