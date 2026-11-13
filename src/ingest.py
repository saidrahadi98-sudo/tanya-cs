"""Build vector database from the faq files (1 file = 1 chunk)"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FAQ_DIR = BASE / "data" / "faq"
DB_DIR = BASE / "db"
COLLECTION = "faq"
MODEL_NAME = "intfloat/multilingual-e5-base"

def load_document(faq_dir=FAQ_DIR):
    """Read every FAQ files to return ids, texts, and metadata lists."""
    ids, texts, metas = [], [], []
    for path in sorted(faq_dir.glob("*.md")):
        if path.name == "facts.md": #Fact information data model not FAQ entry
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = text.splitlines()[0].lstrip("#").strip()
        ids.append(path.name)
        texts.append(text)
        metas.append({"source": path.name, "title": title})
    return ids, texts, metas

def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    ids, texts, metas = load_document()
    print(f"Loaded {len(ids)} FAQ files from {FAQ_DIR}")
    if not ids:
        raise SystemExit("No .md files found. check that data/faq/ has your FAQ files.")

    print(f"Loading model {MODEL_NAME} (the first run downloads about 1 GB)...")
    model = SentenceTransformer(MODEL_NAME)
    # e5 models expect this prefix on documents
    embeddings = model.encode(
        ["passage: " + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)
    col.add(ids=ids, documents=texts, embeddings=embeddings.tolist(), metadatas=metas)
    print(f"stored {col.count()} entries in {DB_DIR}")


if __name__ == "__main__":
    main()