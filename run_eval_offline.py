# eval_rag/run_eval_offline.py
import math, os, sys

# Ajoute le dossier parent au PYTHONPATH pour pouvoir importer rag_bot.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag_bot import rag_bot
from dataset import DATASET
from evaluators_local import (
    correctness, relevance, groundedness, retrieval_relevance
)

def bool_rate(xs):
    if not xs: return 0.0
    return sum(1 for x in xs if x) / len(xs)

def format_bool(val: bool, label: str) -> str:
    """Affiche ✅ si True, ❌ en rouge si False"""
    if val:
        return f"✅ {label}: True"
    else:
        return f"\033[91m❌ {label}: False\033[0m"

def main():
    rows = []
    for ex in DATASET:
        q = ex["inputs"]["question"]
        ref = ex["outputs"]
        out = rag_bot(q)   # {"answer":..., "documents":[...]}

        row = {
            "question": q,
            "answer": out["answer"], 
            "correctness": correctness(ex["inputs"], out, ref),
            "relevance": relevance(ex["inputs"], out),
            "groundedness": groundedness(ex["inputs"], out),
            "retrieval_relevance": retrieval_relevance(ex["inputs"], out),
        }
        rows.append(row)

    # Affichage détaillé
    for r in rows:
        print("\n---")
        print("Q:", r["question"])
        print("Ans:", r["answer"][:500])
        print(format_bool(r["correctness"], "correctness"), "|",
              format_bool(r["relevance"], "relevance"), "|",
              format_bool(r["groundedness"], "groundedness"), "|",
              format_bool(r["retrieval_relevance"], "retrieval_relevance"))

    # Résumé global
    c = bool_rate([r["correctness"] for r in rows])
    rel = bool_rate([r["relevance"] for r in rows])
    grd = bool_rate([r["groundedness"] for r in rows])
    ret = bool_rate([r["retrieval_relevance"] for r in rows])
    print("\n====== Résumé ======")
    print(f"Correctness:          {c:.0%}")
    print(f"Relevance:            {rel:.0%}")
    print(f"Groundedness:         {grd:.0%}")
    print(f"Retrieval relevance:  {ret:.0%}")

if __name__ == "__main__":
    main()
