"""
Shared Chroma collection and embeddings (lazy init).

Uses fastembed + ONNX instead of PyTorch/sentence-transformers to keep
install size and RAM use low (better for Railway and similar hosts).

If you previously indexed with sentence-transformers, delete ./chroma_db
and run: python ingest.py
"""

from __future__ import annotations

import os
from typing import Optional, cast

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "legal_docs"
# Same semantic model family as before; vectors differ from PyTorch ST — re-ingest if upgrading.
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualMiniLMEmbedding(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=MODEL_ID)
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        model = self._get_model()
        return cast(
            Embeddings,
            [
                np.asarray(emb, dtype=np.float32)
                for emb in model.embed(list(input))
            ],
        )


_collection: Optional[chromadb.Collection] = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        emb = MultilingualMiniLMEmbedding()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb,
        )
    return _collection
