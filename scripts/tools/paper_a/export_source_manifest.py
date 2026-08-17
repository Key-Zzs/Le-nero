#!/usr/bin/env python3
"""Export an immutable, read-only source inventory for Paper A datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_real_dataset import build_source_manifest, discover_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="One dataset root or a parent containing datasets.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video-hash", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    datasets = [root] if (root / "meta/info.json").is_file() else discover_datasets(root)
    payload = {
        "schema_version": 1,
        "manifest_type": "paper_a_source_file_inventory",
        "raw_root": str(root),
        "read_only": True,
        "datasets": [build_source_manifest(dataset, hash_video_payloads=not args.no_video_hash) for dataset in datasets],
    }
    payload["manifest_hash"] = __import__("hashlib").sha256(json.dumps(payload["datasets"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(datasets)} source manifest(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

