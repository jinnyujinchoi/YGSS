"""Embedding provider abstractions for RAG evaluation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for OpenAI embedding calls."""

    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "text-embedding-3-small"


class EmbeddingProvider:
    """Base protocol for embedding providers."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text into a float vector."""
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Embed multiple texts preserving input order."""
        return [self.embed_text(text) for text in texts]


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedder for reproducible local evaluation."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    def _hash_token(self, token: str) -> np.ndarray:
        seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dimension, dtype=np.float32)

    @staticmethod
    def _char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> List[str]:
        compact = "".join(ch for ch in text.lower() if not ch.isspace())
        grams: List[str] = []
        for n in range(min_n, max_n + 1):
            if len(compact) < n:
                continue
            for i in range(len(compact) - n + 1):
                grams.append(compact[i : i + n])
        return grams

    def embed_text(self, text: str) -> np.ndarray:
        tokens = [tok for tok in text.lower().split() if tok]
        tokens.extend(self._char_ngrams(text))
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            vec += self._hash_token(token)

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text embedding provider."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not available")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key env: {config.api_key_env}. "
                "Use local-hash provider for offline runs."
            )

        base_url = os.getenv(config.base_url_env) or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = config.model

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=text)
        emb = response.data[0].embedding
        return np.array(emb, dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def create_embedder(
    provider_name: str,
    openai_config: OpenAIEmbeddingConfig,
    dimension: int,
) -> EmbeddingProvider:
    """Create the configured embedding provider with safe fallback."""
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(openai_config)
    if provider_name == "local-hash":
        return LocalHashEmbeddingProvider(dimension=dimension)
    raise ValueError(f"Unsupported embedding provider: {provider_name}")
