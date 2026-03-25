#!/usr/bin/env python3
"""
Upload a local file to POST /api/context/upload on a running crxon_faq_llm API.

Start the server first (e.g. `poetry run fastapi dev src/main.py`, `make debug-server`,
or Docker). A successful upload also needs the normal stack (Redis, KB directory,
embedding model on first use) as in src/main.py lifespan.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import requests


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="POST a real file to /api/context/upload (multipart/form-data).",
    )
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="FILE",
        help="Path to the file (.md, .pdf, .txt); basename must match API filename rules",
    )
    p.add_argument(
        "-f",
        "--file",
        dest="path_alt",
        metavar="FILE",
        default=None,
        help="Same as positional FILE (either is required)",
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API origin without trailing slash (default: %(default)s)",
    )
    p.add_argument(
        "--scope",
        default="global",
        choices=("global", "local", "category"),
        help="Context scope (default: %(default)s)",
    )
    p.add_argument(
        "--persistence",
        default="persistent",
        choices=("persistent", "ephemeral"),
        help="Persistence mode (default: %(default)s)",
    )
    p.add_argument("--session-id", default=None, help="Required when scope=local")
    p.add_argument("--category-id", default=None, help="Required when scope=category")
    p.add_argument("--title", default=None, help="Optional display title")
    p.add_argument("--description", default=None, help="Optional description")
    p.add_argument(
        "--tags",
        default=None,
        help='Optional JSON array of strings, e.g. \'["faq","hr"]\'',
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    file_path = args.path_alt or args.path
    if not file_path:
        parser.error("specify FILE as a positional argument or use --file")

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    url = f"{base}/api/context/upload"

    data: dict[str, str] = {
        "scope": args.scope,
        "persistence": args.persistence,
    }
    if args.session_id is not None:
        data["session_id"] = args.session_id
    if args.category_id is not None:
        data["category_id"] = args.category_id
    if args.title is not None:
        data["title"] = args.title
    if args.description is not None:
        data["description"] = args.description
    if args.tags is not None:
        data["tags"] = args.tags

    mime, _ = mimetypes.guess_type(path.name)
    content_type = mime or "application/octet-stream"

    try:
        with path.open("rb") as f:
            files = {"file": (path.name, f, content_type)}
            resp = requests.post(url, files=files, data=data, timeout=args.timeout)
    except requests.RequestException as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"HTTP {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except json.JSONDecodeError:
        print(resp.text)

    return 0 if resp.status_code == 201 else 1


if __name__ == "__main__":
    raise SystemExit(main())
