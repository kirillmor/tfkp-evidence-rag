#!/usr/bin/env python3
"""Собирает локальный гибридный индекс из корпуса ТФКП."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_core import load_json, read_jsonl, tokenize


def validate_corpus(rows: list[dict[str, object]]) -> None:
    ids: set[str] = set()
    for row in rows:
        required = {"block_id", "page", "section", "text"}
        if required - set(row):
            raise ValueError(f"Блок не содержит поля: {required - set(row)}")
        block_id = row["block_id"]
        if not isinstance(block_id, str) or block_id in ids:
            raise ValueError("block_id должен быть непустым и уникальным")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Пустой текст блока {block_id}")
        ids.add(block_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    documents = read_jsonl(args.corpus)
    validate_corpus(documents)
    config = load_json(args.config)
    tokenized = [tokenize(str(document["text"])) for document in documents]
    if any(not tokens for tokens in tokenized):
        raise ValueError("После токенизации обнаружен пустой блок")

    bm25 = BM25Okapi(tokenized)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    char_matrix = vectorizer.fit_transform([str(document["text"]) for document in documents])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "documents": documents,
            "bm25": bm25,
            "vectorizer": vectorizer,
            "char_matrix": char_matrix,
            "config": config,
        },
        output / "index.joblib",
    )
    metadata = {
        "documents": len(documents),
        "corpus": str(Path(args.corpus).resolve()),
        "retrieval": config["retrieval"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Индекс из {len(documents)} страниц сохранён в {output}")


if __name__ == "__main__":
    main()
