"""Общие функции для построения индекса и запуска доказательного RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Некорректный JSONL в строке {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Строка {line_number} должна содержать JSON-объект")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def tokenize(text: str) -> list[str]:
    """Нормализует русский, латинский и математические буквенные токены."""
    return TOKEN_RE.findall(text.lower().replace("ё", "е"))


def question_text(question: dict[str, Any], include_options: bool) -> str:
    text = str(question["question"])
    if include_options:
        text += "\n" + "\n".join(
            f"{option['id']}: {option['text']}" for option in question["options"]
        )
    return text


def reciprocal_rank_scores(scores: np.ndarray) -> np.ndarray:
    """Переводит несопоставимые шкалы двух поисков в устойчивые ранги."""
    result = np.zeros(len(scores), dtype=float)
    for rank, index in enumerate(np.argsort(-scores), 1):
        result[index] = 1.0 / (60 + rank)
    return result


def safe_rationale(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:1200]


def validate_question(row: dict[str, Any]) -> None:
    required = {"id", "question", "options"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Вопрос не содержит полей: {', '.join(sorted(missing))}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("ID вопроса должен быть непустой строкой")
    if not isinstance(row["options"], list) or not 4 <= len(row["options"]) <= 6:
        raise ValueError(f"Вопрос {row['id']}: требуется от 4 до 6 вариантов")
    option_ids = [item.get("id") for item in row["options"] if isinstance(item, dict)]
    if len(option_ids) != len(row["options"]) or len(option_ids) != len(set(option_ids)):
        raise ValueError(f"Вопрос {row['id']}: ID вариантов должны быть уникальны")


def validate_questions(rows: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for row in rows:
        validate_question(row)
        if row["id"] in ids:
            raise ValueError(f"Повторяющийся ID вопроса: {row['id']}")
        ids.add(row["id"])
