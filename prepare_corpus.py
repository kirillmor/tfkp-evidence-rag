#!/usr/bin/env python3
"""Преобразует PDF учебника в воспроизводимый постраничный корпус JSONL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz

from rag_core import write_jsonl


SECTION_RE = re.compile(r"(?:§\s*\d+(?:\.\d+)?|Глава\s+\d+)", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Убирает технические переносы PDF, не меняя содержание формул."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_corpus(pdf_path: str | Path) -> list[dict[str, object]]:
    document = fitz.open(pdf_path)
    section = "Не определён"
    rows: list[dict[str, object]] = []
    try:
        for page_number, page in enumerate(document, 1):
            text = clean_text(page.get_text("text"))
            matches = list(SECTION_RE.finditer(text))
            if matches:
                section = matches[-1].group(0).replace("  ", " ")
            rows.append(
                {
                    "block_id": f"p_{page_number:04d}",
                    "page": page_number,
                    "section": section,
                    "text": text,
                }
            )
    finally:
        document.close()
    if not rows or not any(row["text"] for row in rows):
        raise ValueError("В PDF не извлечён текст")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Путь к скачанному учебнику")
    parser.add_argument("--output", default="data/generated/corpus.jsonl")
    args = parser.parse_args()
    rows = extract_corpus(args.pdf)
    write_jsonl(args.output, rows)
    print(f"Записано {len(rows)} блоков в {args.output}")


if __name__ == "__main__":
    main()
