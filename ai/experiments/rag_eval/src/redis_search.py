"""Redis SCAN equivalent search and dump-based fallback search."""

from __future__ import annotations

import json
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]


CHAT_DUMMY_INSERT = "INSERT INTO `chat_dummy`(`term_id`, `question`, `answer`) VALUES"


@dataclass
class QARecord:
    """Source row from chat_dummy."""

    answer_id: int
    term_id: int
    question: str
    answer: str


@dataclass
class SearchResult:
    """Search output equivalent to Java SearchResultDto enriched with answer text."""

    prefix: str
    term_id: int
    answer_id: int
    vector_type: str
    similarity: float
    answer: str


@dataclass
class DumpVectorRow:
    """Embedding dump row for local evaluations."""

    prefix: str
    term_id: int
    answer_id: int
    vector_type: str
    text: str
    embedding: np.ndarray


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity with safe zero guards."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_chat_dummy_sql(sql_path: Path) -> Dict[int, QARecord]:
    """Parse chat_dummy insert block and return id-indexed records."""
    text = sql_path.read_text(encoding="utf-8")
    start = text.find(CHAT_DUMMY_INSERT)
    if start == -1:
        raise ValueError(f"Cannot find chat_dummy insert block in: {sql_path}")

    block = text[start:]
    end = block.find(";\n\nINSERT INTO")
    if end != -1:
        block = block[: end + 1]

    pattern = re.compile(
        r'^\((\d+),\s*"((?:[^"\\]|\\.)*)",\s*"((?:[^"\\]|\\.)*)"\),?$',
        re.MULTILINE,
    )

    rows: Dict[int, QARecord] = {}
    row_id = 0
    for match in pattern.finditer(block):
        row_id += 1
        term_id = int(match.group(1))
        question = match.group(2).replace("\\\"", "\"").replace("\\\\", "\\")
        answer = match.group(3).replace("\\\"", "\"").replace("\\\\", "\\")
        rows[row_id] = QARecord(
            answer_id=row_id,
            term_id=term_id,
            question=question,
            answer=answer,
        )

    if not rows:
        raise ValueError(f"No chat_dummy rows parsed from: {sql_path}")
    return rows


def _decode_java_float_array(value: bytes) -> np.ndarray:
    """Decode Java ByteBuffer float32 array (big endian) from Redis list values."""
    if len(value) % 4 != 0:
        return np.array([], dtype=np.float32)
    size = len(value) // 4
    floats = struct.unpack(f">{size}f", value)
    return np.array(floats, dtype=np.float32)


class DumpVectorSearcher:
    """In-memory searcher that mimics VectorRepository.searchAllPrefixes."""

    def __init__(self, rows: Sequence[DumpVectorRow], answer_map: Dict[int, QARecord]) -> None:
        self.rows = list(rows)
        self.answer_map = answer_map

    @classmethod
    def from_jsonl(cls, path: Path, answer_map: Dict[int, QARecord]) -> "DumpVectorSearcher":
        """Build from JSONL embedding dump."""
        rows: List[DumpVectorRow] = []
        if not path.exists():
            raise FileNotFoundError(f"Embedding dump not found: {path}")

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                DumpVectorRow(
                    prefix=obj["prefix"],
                    term_id=int(obj["term_id"]),
                    answer_id=int(obj["answer_id"]),
                    vector_type=obj.get("vector_type", "Q"),
                    text=obj.get("text", ""),
                    embedding=np.array(obj["embedding"], dtype=np.float32),
                )
            )
        return cls(rows=rows, answer_map=answer_map)

    @classmethod
    def from_sql_records(
        cls,
        records: Dict[int, QARecord],
        embed_text_fn,
    ) -> "DumpVectorSearcher":
        """Build dump-style corpus by embedding chat_dummy questions."""
        rows: List[DumpVectorRow] = []
        ids = sorted(records.keys())
        questions = [records[idx].question for idx in ids]
        vectors = embed_text_fn(questions)

        for answer_id, vec in zip(ids, vectors):
            row = records[answer_id]
            rows.append(
                DumpVectorRow(
                    prefix="term",
                    term_id=row.term_id,
                    answer_id=row.answer_id,
                    vector_type="Q",
                    text=row.question,
                    embedding=vec,
                )
            )
        return cls(rows=rows, answer_map=records)

    def search(self, query_vector: np.ndarray, k: int, min_similarity: float) -> List[SearchResult]:
        """Search vectors using the same threshold + top-k logic as Java code."""
        scored: List[SearchResult] = []
        for row in self.rows:
            similarity = cosine_similarity(query_vector, row.embedding)
            if similarity < min_similarity:
                continue

            answer_text = self.answer_map.get(row.answer_id)
            scored.append(
                SearchResult(
                    prefix=row.prefix,
                    term_id=row.term_id,
                    answer_id=row.answer_id,
                    vector_type=row.vector_type,
                    similarity=similarity,
                    answer=answer_text.answer if answer_text else "",
                )
            )

        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:k]


class RedisVectorSearcher:
    """Redis SCAN searcher with production safety controls."""

    def __init__(
        self,
        answer_map: Dict[int, QARecord],
        scan_count: int = 200,
        max_keys: int = 50000,
        timeout_seconds: int = 20,
        key_pattern: str = "*",
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is not available")
        self.answer_map = answer_map
        self.scan_count = scan_count
        self.max_keys = max_keys
        self.timeout_seconds = timeout_seconds
        self.key_pattern = key_pattern

        self.client = self._connect()

    def _connect(self):
        host = os.getenv("RAG_EVAL_REDIS_HOST", "")
        if not host:
            raise RuntimeError("RAG_EVAL_REDIS_HOST is required in redis mode")

        enable_prod = os.getenv("ENABLE_PROD_REDIS", "false").lower() == "true"
        prod_hosts = {
            item.strip() for item in os.getenv("PROD_REDIS_HOSTS", "").split(",") if item.strip()
        }

        is_prod_host = host in prod_hosts
        if is_prod_host and not enable_prod:
            raise RuntimeError(
                "Production Redis host blocked. Set ENABLE_PROD_REDIS=true to proceed."
            )

        if enable_prod:
            print("[safety] ENABLE_PROD_REDIS=true detected.")
            print("[safety] Use read-only credentials only before continuing.")
            proceed = os.getenv("PROCEED_PROD_REDIS", "").upper()
            if proceed != "YES":
                raise RuntimeError(
                    "Set PROCEED_PROD_REDIS=YES after manual confirmation to continue."
                )

        return redis.Redis(
            host=host,
            port=int(os.getenv("RAG_EVAL_REDIS_PORT", "6379")),
            db=int(os.getenv("RAG_EVAL_REDIS_DB", "0")),
            username=os.getenv("RAG_EVAL_REDIS_USERNAME") or None,
            password=os.getenv("RAG_EVAL_REDIS_PASSWORD") or None,
            ssl=os.getenv("RAG_EVAL_REDIS_SSL", "false").lower() == "true",
            decode_responses=False,
            socket_timeout=self.timeout_seconds,
        )

    def search(self, query_vector: np.ndarray, k: int, min_similarity: float) -> List[SearchResult]:
        """Replicate VectorRepository.searchAllPrefixes using SCAN only."""
        cursor = 0
        scanned = 0
        started = time.time()
        scored: List[SearchResult] = []

        while True:
            if scanned >= self.max_keys:
                break
            if time.time() - started > self.timeout_seconds:
                break

            cursor, keys = self.client.scan(
                cursor=cursor,
                match=self.key_pattern,
                count=self.scan_count,
            )

            for key in keys:
                scanned += 1
                if scanned > self.max_keys:
                    break

                key_str = key.decode("utf-8", errors="ignore")
                key_type = self.client.type(key)
                key_type_str = key_type.decode("utf-8", errors="ignore")
                if key_type_str.lower() != "list":
                    continue

                parts = key_str.split(":")
                if len(parts) < 4:
                    continue

                prefix = parts[0]
                try:
                    term_id = int(parts[1])
                    answer_id = int(parts[2])
                except ValueError:
                    continue
                vector_type = parts[3].upper()

                embeddings = self.client.lrange(key, 0, -1)
                for value in embeddings:
                    vector = _decode_java_float_array(value)
                    if vector.size == 0:
                        continue
                    similarity = cosine_similarity(query_vector, vector)
                    if similarity < min_similarity:
                        continue

                    answer = self.answer_map.get(answer_id)
                    scored.append(
                        SearchResult(
                            prefix=prefix,
                            term_id=term_id,
                            answer_id=answer_id,
                            vector_type=vector_type,
                            similarity=similarity,
                            answer=answer.answer if answer else "",
                        )
                    )

            if cursor == 0:
                break

        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:k]
