#!/usr/bin/env python3
"""Скачивает единственный разрешённый источник корпуса и фиксирует его хеш."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://new.mipt.ru/upload/mathematics/study/uchebniki/"
    "%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_"
    "%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/tfkp_2025.pdf")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(args.url, headers={"User-Agent": "tfkp-rag-homework/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload.startswith(b"%PDF"):
        raise RuntimeError("Скачанный файл не похож на PDF")
    output.write_bytes(payload)
    metadata = {
        "source_url": args.url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    output.with_suffix(output.suffix + ".sha256.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Скачан {output} ({metadata['bytes']} байт, sha256={metadata['sha256']})")


if __name__ == "__main__":
    main()
