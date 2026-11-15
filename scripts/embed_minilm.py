"""Lab 8 comparison: embed the same passages with the transformer all-MiniLM-L6-v2.

Needs PyTorch, so it is NOT a project dependency (the 512 MB server could not hold it):

    uv run --with sentence-transformers python scripts/embed_minilm.py data/index.pkl data/index-minilm.emb
    python scripts/eval_lab08.py data/index.pkl data/index-minilm.emb
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findex.semantic import embed_index
from findex.store import load

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLM:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL, device="cpu")

    def encode(self, sentences: list[str]) -> Any:
        return self.model.encode(sentences, batch_size=64, convert_to_numpy=True)


def main() -> None:
    index = load(Path(sys.argv[1]))
    t0 = time.perf_counter()
    semantic = embed_index(index, MiniLM(), model_name=MODEL, batch=1024)
    semantic.save(Path(sys.argv[2]))
    print(f"{semantic.vectors.shape} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
