"""Run test cases against a running server and report metrics.

Usage:  python eval/run_eval.py [--url http://127.0.0.1:8000] [--cases eval/testcases.jsonl]

Test case fields: id, lang, question, answerable, gold_source, keywords (any-of, case-insensitive).
Metrics
  retrieval hit@k      gold_source appears in returned sources          (answerable cases)
  keyword recall       fraction of cases whose answer has >=1 keyword   (factual-correctness proxy)
  refusal accuracy     answerable -> grounded, unanswerable -> not grounded
Review results.csv by hand for groundedness and write the failure analysis from the misses.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--cases", default=str(Path(__file__).parent / "testcases.jsonl"))
ap.add_argument("--out", default=str(Path(__file__).parent / "results.csv"))
args = ap.parse_args()

cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
rows = []
with httpx.Client(timeout=300) as client:
    for c in cases:
        r = client.post(f"{args.url}/api/ask", json={"question": c["question"], "lang": c["lang"]}).json()
        # For translated languages, also score the intermediate English answer: Hindi spellings of
        # chemical names vary after back-translation, which would otherwise count as a wrong answer.
        ans = (r["answer"] + " " + (r.get("english_answer") or "")).lower()
        hit = c["gold_source"] in {s["source"] for s in r["sources"]} if c["answerable"] else None
        kw = any(k.lower() in ans for k in c["keywords"]) if c["answerable"] and c["keywords"] else None
        rows.append({
            "id": c["id"], "lang": c["lang"], "question": c["question"], "answerable": c["answerable"],
            "grounded": r["grounded"], "refusal_correct": r["grounded"] == c["answerable"],
            "retrieval_hit": hit, "keyword_hit": kw, "best_score": r["best_score"],
            "latency_ms": sum(r["timings"].values()),
            "unsupported_numbers": " ".join(r.get("unsupported_numbers") or []),
            "english_question": r.get("english_question") or "", "answer": r["answer"].replace("\n", " "),
        })
        print(f"#{c['id']:>3} {c['lang']} grounded={r['grounded']!s:5} hit={hit} kw={kw}  {c['question'][:50]}")

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)


def rate(key, subset=rows):
    v = [x[key] for x in subset if x[key] is not None]
    return f"{sum(v)}/{len(v)} = {sum(v) / len(v):.0%}" if v else "n/a"


print("\n== Summary ==")
print("cases:", len(rows))
print("retrieval hit@k :", rate("retrieval_hit"))
print("keyword recall  :", rate("keyword_hit"))
print("refusal accuracy:", rate("refusal_correct"))
for lang in sorted({x["lang"] for x in rows}):
    sub = [x for x in rows if x["lang"] == lang]
    print(f"  [{lang}] refusal {rate('refusal_correct', sub)}  keyword {rate('keyword_hit', sub)}")
print("median latency  :", int(statistics.median(x["latency_ms"] for x in rows)), "ms")
print("wrote", args.out)
