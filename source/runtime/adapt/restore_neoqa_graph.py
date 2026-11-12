"""Restore a portable, pristine NeoQA snapshot into a new native Chroma store."""
import argparse
import json
import os
from pathlib import Path


def restore(snapshot, target):
    import chromadb
    import numpy as np
    snapshot, target = Path(snapshot).resolve(), Path(target).resolve()
    if os.name == "nt" and not str(target).isascii():
        raise ValueError("This Windows Chroma/HNSW runtime requires an ASCII-only target path; use e.g. C:/rag_runs/food/chroma_data. The source snapshot path may contain Chinese characters.")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Target must be empty; use a separate run directory")
    target.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target))
    stats = {}
    for name in ("entity", "chunk", "relation", "connection"):
        data = json.loads((snapshot / f"{name}.json").read_text(encoding="utf-8"))
        vectors = np.load(snapshot / f"{name}_vectors.npz")["vectors"]
        assert len(data["ids"]) == vectors.shape[0] and np.isfinite(vectors).all()
        collection = client.get_or_create_collection(name=name)
        for start in range(0, len(data["ids"]), 250):
            end = start + 250
            collection.add(ids=data["ids"][start:end], documents=data["documents"][start:end],
                           metadatas=data["metadatas"][start:end], embeddings=vectors[start:end])
        assert collection.count() == len(data["ids"])
        stats[name] = collection.count()
    client.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.snapshot, args.target)))
