from __future__ import annotations

from pathlib import Path

import yaml

from src.cross_encoder import CrossEncoderCandidate, load_candidates_with_status


def test_all_candidates_loadable() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    candidates = [CrossEncoderCandidate(**item) for item in cfg["models"]["cross_encoders"]["candidates"]]
    loaded, status = load_candidates_with_status(candidates)

    failures = [
        f"{name}: {row.get('error', 'unknown error')}"
        for name, row in status.items()
        if row.get("status") != "ok"
    ]
    assert len(loaded) == len(candidates), "\n".join(failures)
