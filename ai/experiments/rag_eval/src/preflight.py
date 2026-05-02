"""Preflight gate for measurement validity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List

from .cross_encoder import CrossEncoderCandidate, load_candidates_with_status
from .embedder import OpenAIEmbeddingConfig, create_embedder
from .redis_search import parse_chat_dummy_sql

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]


@dataclass
class PreflightResult:
    passed: bool
    checks: Dict[str, object]
    models_loaded: List[str]
    model_status_path: str
    report_path: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check_redis(cfg: Dict) -> Dict[str, object]:
    if redis is None:
        return {"ok": False, "error": "redis package is not available"}

    host = "127.0.0.1"
    port = 6379
    db = 0
    try:
        client = redis.Redis(host=host, port=port, db=db, socket_timeout=int(cfg["redis"]["timeout_seconds"]))
        cursor, keys = client.scan(cursor=0, match=str(cfg["redis"].get("key_pattern", "*")), count=5)
        del cursor
        return {
            "ok": len(keys) > 0,
            "host": host,
            "port": port,
            "db": db,
            "key_count_sample": len(keys),
            "error": "" if len(keys) > 0 else "redis scan returned zero keys",
        }
    except Exception as exc:
        return {"ok": False, "host": host, "port": port, "db": db, "error": str(exc)}


def run_preflight(cfg: Dict, root: Path) -> PreflightResult:
    results_dir = root / cfg["results"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, object] = {}

    embedding_provider = str(cfg["embedding"]["provider"])
    checks["embedding_provider"] = {
        "ok": embedding_provider == "openai",
        "value": embedding_provider,
        "expected": "openai",
    }

    openai_cfg = OpenAIEmbeddingConfig(
        api_key_env=cfg["openai"]["api_key_env"],
        base_url_env=cfg["openai"]["base_url_env"],
        model=cfg["openai"]["embedding_model"],
    )
    try:
        embedder = create_embedder(provider_name=embedding_provider, openai_config=openai_cfg, dimension=1536)
        _ = embedder.embed_text("preflight ping")
        checks["openai_embedding_call"] = {"ok": True, "model": openai_cfg.model}
    except Exception as exc:
        checks["openai_embedding_call"] = {"ok": False, "model": openai_cfg.model, "error": str(exc)}

    source_mode = str(cfg["data"].get("source_mode", "redis"))
    if source_mode == "redis":
        checks["redis_scan"] = _check_redis(cfg)
    else:
        checks["redis_scan"] = {"ok": True, "skipped": True, "reason": f"source_mode={source_mode}, Redis not required"}

    eval_set_path = (root / cfg["data"]["eval_set_path"]).resolve()
    try:
        payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
        enough_items = len(payload) >= 30
        has_answers = all(bool(item.get("correct_answer_ids")) for item in payload)
        checks["eval_set"] = {
            "ok": enough_items and has_answers,
            "count": len(payload),
            "count_requirement": ">=30",
            "all_have_correct_answer_ids": has_answers,
        }
    except Exception as exc:
        checks["eval_set"] = {"ok": False, "error": str(exc)}

    # SQL parse is required for measurement input integrity.
    try:
        _ = parse_chat_dummy_sql((root / cfg["data"]["chat_dummy_sql_path"]).resolve())
        checks["sql_source"] = {"ok": True}
    except Exception as exc:
        checks["sql_source"] = {"ok": False, "error": str(exc)}

    candidates = [CrossEncoderCandidate(**item) for item in cfg["models"]["cross_encoders"]["candidates"]]
    loaded_models, model_status = load_candidates_with_status(candidates)

    model_status_path = results_dir / "model_load_status.json"
    model_status_path.write_text(json.dumps(model_status, ensure_ascii=False, indent=2), encoding="utf-8")

    checks["cross_encoder_models"] = {
        "ok": len(loaded_models) == len(candidates),
        "loaded": sorted(loaded_models.keys()),
        "required": [item.name for item in candidates],
    }

    passed = all(bool(item.get("ok")) for item in checks.values() if isinstance(item, dict) and "ok" in item)

    report = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "checks": checks,
        "model_load_status_file": str(model_status_path.relative_to(root)),
    }
    report_path = results_dir / f"preflight_{_timestamp()}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return PreflightResult(
        passed=passed,
        checks=checks,
        models_loaded=sorted(loaded_models.keys()),
        model_status_path=str(model_status_path),
        report_path=str(report_path),
    )
