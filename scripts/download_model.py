#!/usr/bin/env python
"""Download model weights for local/API deployment."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HuggingFace Hub repo id")
    parser.add_argument("--filename", required=True, help="Remote checkpoint filename")
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/model.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to download model weights.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=args.repo_id, filename=args.filename)
    args.output.write_bytes(Path(downloaded).read_bytes())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
