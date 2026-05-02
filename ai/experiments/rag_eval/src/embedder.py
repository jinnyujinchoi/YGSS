"""Embedding provider abstractions for RAG evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

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
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text embedding provider."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not available")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key env: {config.api_key_env}")

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
    """Create the configured embedding provider with strict mode."""
    del dimension
    if provider_name != "openai":
        raise RuntimeError(f"Unsupported embedding provider for measurement mode: {provider_name}")
    return OpenAIEmbeddingProvider(openai_config)
