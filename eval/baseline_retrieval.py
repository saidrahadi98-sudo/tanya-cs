"""Retrieval check on the whole test set: hit@k per category.

Usage (from the TanyaCS folder):
    python eval/baseline_retrieval.py             # saves eval/results/baseline.csv
    python eval/baseline_retrieval.py normalized  # saves eval/results/normalized.csv
"""
import csv
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEST_SET = BASE / "data" / "test_set.csv"
DB_DIR = BASE / "db"
RESULTS_DIR = BASE / "eval" / "results"
MODEL_NAME = "intfloat/multilingual-e5-base"
K = 4
IN_SCOPE = ["formal", "slang_abbrev", "typo", "code_mixed_en", "banjar_mixed"]


def load_test_set(path=TEST_SET):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(rows, retrieved):
    """retrieved[i] = (list of source names, list of distances) for rows[i]."""
    results = []
    for row, (sources, dists) in zip(rows, retrieved):
        expected = row["expected_faq"]
        rank = sources.index(expected) + 1 if expected in sources else None
        results.append({**row, "rank": rank, "best": dists[0], "sources": sources})
    return results


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def report(results, label, results_dir=RESULTS_DIR):
    hk_label = f"hit@{K}"
    print(f"\n=== Retrieval results ({label}, k={K}) ===")
    print(f"{'category':<15}{'n':>4}{'hit@1':>9}{hk_label:>9}")
    total = h1 = hk = 0
    for cat in IN_SCOPE:
        rs = [r for r in results if r["category"] == cat]
        a = sum(1 for r in rs if r["rank"] == 1)
        b = sum(1 for r in rs if r["rank"] is not None)
        print(f"{cat:<15}{len(rs):>4}{pct(a, len(rs)):>9}{pct(b, len(rs)):>9}")
        total += len(rs)
        h1 += a
        hk += b
    print(f"{'ALL in-scope':<15}{total:>4}{pct(h1, total):>9}{pct(hk, total):>9}")

    in_scope = [r for r in results if r["category"] in IN_SCOPE]
    oos = [r for r in results if r["category"] == "out_of_scope"]
    hits = [r["best"] for r in in_scope if r["rank"] is not None]
    misses = [r["best"] for r in in_scope if r["rank"] is None]
    print("\nMedian best distance (lower = closer):")
    for name, vals in (("in-scope hits", hits), ("in-scope misses", misses),
                       ("out-of-scope", [r["best"] for r in oos])):
        if vals:
            print(f"  {name:<16}{statistics.median(vals):.3f}  (n={len(vals)})")

    print("\nIf the bot refuses whenever the best distance is above a cutoff:")
    print(f"{'cutoff':>7}{'in-scope kept':>15}{'kept and right':>16}{'out-of-scope refused':>22}")
    for i in range(14):
        t = 0.30 + 0.02 * i
        kept = [r for r in in_scope if r["best"] <= t]
        right = [r for r in kept if r["rank"] is not None]
        refused = [r for r in oos if r["best"] > t]
        print(f"{t:>7.2f}{pct(len(kept), len(in_scope)):>15}{pct(len(right), len(in_scope)):>16}"
              f"{pct(len(refused), len(oos)):>22}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "question", "expected_faq", "rank", "best_distance"]
                   + [f"top{i}" for i in range(1, K + 1)])
        for r in results:
            w.writerow([r["id"], r["category"], r["question"], r["expected_faq"],
                        r["rank"] or "", f"{r['best']:.3f}"] + r["sources"])
    print(f"\nPer-question details saved to {out_path}")


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    rows = load_test_set()
    print(f"Loaded {len(rows)} test questions from {TEST_SET}")

    col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection("faq")
    known = set(col.get()["ids"])
    unknown = sorted({r["expected_faq"] for r in rows} - known - {"NONE"})
    if unknown:
        print("WARNING: these labels in test_set.csv are not in the database "
              "(renamed file? run src/ingest.py again):")
        for name in unknown:
            print("  -", name)

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        ["query: " + r["question"] for r in rows],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    res = col.query(query_embeddings=embeddings.tolist(), n_results=K)
    retrieved = [([m["source"] for m in res["metadatas"][i]], res["distances"][i])
                 for i in range(len(rows))]
    report(evaluate(rows, retrieved), label)


if __name__ == "__main__":
    main()