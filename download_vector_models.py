#!/usr/bin/env python3
"""Download local retrieval models used by the paper knowledge base."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent


def resolve_model_root(output_dir: Optional[Path] = None) -> Path:
    if output_dir:
        return output_dir
    val = os.getenv("PAPER_MODEL_ROOT")
    if val:
        return Path(val)
    return BASE_DIR / "models"


DEFAULT_MODELS = {
    "embedding": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download embedding and reranker model snapshots."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["all"],
        metavar="MODEL",
        help="embedding, reranker, or all (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for local model snapshots; defaults to PAPER_MODEL_ROOT.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent download workers (default: 4).",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing an optional access token.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the model registry without downloading.",
    )
    return parser.parse_args()


def selected_models(names: list[str]) -> list[tuple[str, str]]:
    unknown = sorted(set(names) - set(DEFAULT_MODELS) - {"all"})
    if unknown:
        raise ValueError(
            "unknown model(s): {}. Expected embedding, reranker, or all".format(
                ", ".join(unknown)
            )
        )
    if "all" in names:
        if len(names) != 1:
            raise ValueError("'all' cannot be combined with individual model names")
        names = list(DEFAULT_MODELS)
    return [(name, DEFAULT_MODELS[name]) for name in names]


def require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: "
            "python -m pip install --upgrade huggingface_hub"
        ) from exc
    return snapshot_download


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    manifest = {
        "schema_version": 1,
        "provider": "huggingface",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "models": records,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


def main() -> int:
    args = parse_args()
    try:
        models = selected_models(args.models)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for name, repo_id in models:
            print(f"{name:10s} {repo_id}")
        return 0

    if args.max_workers < 1:
        print("error: --max-workers must be >= 1", file=sys.stderr)
        return 2

    try:
        snapshot_download = require_huggingface_hub()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = resolve_model_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(args.token_env) or None
    records: list[dict[str, object]] = []

    for name, repo_id in models:
        local_dir = output_dir / name
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        resolved_path = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=str(local_dir),
            token=token,
            max_workers=args.max_workers,
        )
        records.append(
            {
                "name": name,
                "repo_id": repo_id,
                "revision": args.revision or "default",
                "path": str(Path(resolved_path).resolve()),
            }
        )

    write_manifest(output_dir, records)
    print("All requested models downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
