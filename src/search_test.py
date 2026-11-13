"""Quick check: ask a few questions and print the 4 closest FAQ files."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
model = SentenceTransformer("intfloat/multilingual-e5-base")
col = chromadb.PersistentClient(path=str(BASE / "db")).get_collection("faq")

QUESTIONS = [
    "min, bayar tagihan bisa lewat apa aja sih?",
    "kok internet gue lemot banget ya min?",
    "Ulun handak berhenti langganan, kayapa caranya?",
    "Rekomendasi film bagus dong",
]

for q in QUESTIONS:
    emb = model.encode(["query: "+ q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=4)
    print("\n" + q)
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  {dist:.3f}  {meta['source']}")