#!/usr/bin/env python3
"""Генерирует открытый и авторский наборы вопросов из версионируемых описаний."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_specs import DEV_ITEMS, OWN_ITEMS
from rag_core import write_jsonl


def convert(items: list[dict]) -> tuple[list[dict], list[dict]]:
    questions, gold = [], []
    for source in items:
        questions.append({key: source[key] for key in ("id", "question", "options")})
        abstain = not source["answer_ids"]
        gold.append(
            {
                "id": source["id"],
                "category": source["category"],
                "answer_ids": source["answer_ids"],
                "abstain": abstain,
                "acceptable_citation_sets": [[f"p_{page:04d}" for page in source["pages"]]],
                "rationale": source["rationale"],
                **({"missing_information": source["missing_information"]} if abstain else {}),
            }
        )
    return questions, gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    output = Path(args.output_dir)
    for prefix, items in (("dev", DEV_ITEMS), ("own", OWN_ITEMS)):
        questions, gold = convert(items)
        write_jsonl(output / f"{prefix}_questions.jsonl", questions)
        write_jsonl(output / f"{prefix}_gold.jsonl", gold)
        print(f"Созданы {prefix}_questions.jsonl и {prefix}_gold.jsonl ({len(items)} вопросов)")


if __name__ == "__main__":
    main()
