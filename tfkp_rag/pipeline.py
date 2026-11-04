#!/usr/bin/env python3
"""Отвечает на вопросы по учебнику с проверяемыми ссылками на страницы корпуса."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from openai import OpenAI

from .core import (
    load_json,
    question_text,
    read_jsonl,
    reciprocal_rank_scores,
    safe_rationale,
    tokenize,
    validate_questions,
    write_jsonl,
)


def search(index: dict[str, Any], query: str, use_char: bool) -> np.ndarray:
    bm25_scores = np.asarray(index["bm25"].get_scores(tokenize(query)), dtype=float)
    if not use_char:
        return bm25_scores
    char_query = index["vectorizer"].transform([query])
    char_scores = (index["char_matrix"] @ char_query.T).toarray().ravel()
    settings = index["config"]["retrieval"]
    return (
        settings["bm25_weight"] * reciprocal_rank_scores(bm25_scores)
        + settings["char_tfidf_weight"] * reciprocal_rank_scores(char_scores)
    )


def select_context(
    index: dict[str, Any], query: str, mode: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if mode == "no-context":
        return [], []
    use_char = mode not in {"single-rag", "no-char"}
    scores = search(index, query, use_char=use_char)
    ranked = list(np.argsort(-scores))
    settings = index["config"]["retrieval"]
    top_count = 4 if mode == "single-rag" else int(settings["top_pages"])
    primary = ranked[:top_count]
    chosen: list[int] = []
    if mode == "single-rag":
        chosen = primary
    else:
        radius = int(settings["neighbor_radius"])
        for index_position in primary:
            for neighbor in range(index_position - radius, index_position + radius + 1):
                if 0 <= neighbor < len(index["documents"]) and neighbor not in chosen:
                    chosen.append(neighbor)
    documents: list[dict[str, Any]] = []
    total_chars = 0
    max_chars = int(settings["max_context_chars"])
    for index_position in chosen:
        document = index["documents"][index_position]
        text = str(document["text"])
        # Последний блок не обрезается посередине, если он целиком не помещается.
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining >= 500:
                document = {**document, "text": text[:remaining]}
                documents.append(document)
            break
        documents.append(document)
        total_chars += len(text)
    return documents, [str(index["documents"][position]["block_id"]) for position in chosen]


def make_prompt(question: dict[str, Any], context: list[dict[str, Any]], retry: bool) -> str:
    option_lines = "\n".join(f"{item['id']}. {item['text']}" for item in question["options"])
    context_text = "\n\n".join(
        f"[{item['block_id']}; PDF стр. {item['page']}]\n{item['text']}" for item in context
    )
    source_rule = (
        "Используй только приведённые фрагменты учебника. "
        "Если данных недостаточно для единственного ответа, верни abstain=true. "
        "При abstain=true answer_ids и citations обязаны быть пустыми. "
        "При abstain=false выбери все и только правильные ID и укажи от 1 до 4 ID фрагментов, "
        "которые подтверждают вывод. Не добавляй знания из памяти."
    )
    if not context:
        source_rule = (
            "Контекст намеренно не предоставлен для эксперимента. Выбери варианты по своим знаниям; "
            "citations верни пустым списком."
        )
    retry_rule = "\nПредыдущий ответ не прошёл проверку формата. Верни только корректный JSON." if retry else ""
    return (
        "Ты отвечаешь на тест по теории функций комплексного переменного.\n"
        f"{source_rule}\n"
        "Верни только JSON без Markdown: "
        '{"answer_ids":["A"],"abstain":false,"rationale":"краткое обоснование","citations":["p_0001"]}.\n'
        f"Вопрос: {question['question']}\nВарианты:\n{option_lines}\n"
        f"Фрагменты:\n{context_text}{retry_rule}"
    )


def prompt_token_upper_bound(prompt: str) -> int:
    """Даёт безопасную верхнюю оценку токенов без вызова внешнего токенизатора.

    Число UTF-8-байтов не меньше числа byte-level токенов. Запас покрывает
    служебные токены chat template, поэтому проверка выполняется до запроса.
    """
    return len(prompt.encode("utf-8")) + 512


def fit_context_to_limits(
    question: dict[str, Any], context: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Оставляет максимум доказательств, не выходя за лимит одного запроса."""
    limits = config["limits"]
    max_prompt_tokens = limits["max_single_request_tokens"] - config["model"]["max_tokens"]
    if prompt_token_upper_bound(make_prompt(question, [], retry=True)) > max_prompt_tokens:
        raise ValueError(f"Вопрос {question['id']} слишком длинный для лимита одного запроса")

    selected: list[dict[str, Any]] = []
    for document in context:
        candidate = selected + [document]
        if prompt_token_upper_bound(make_prompt(question, candidate, retry=True)) <= max_prompt_tokens:
            selected = candidate
            continue

        # Сохраняем начало последнего фрагмента, если оно ещё помещается.
        text = str(document["text"])
        low, high, best = 0, len(text), 0
        while low <= high:
            middle = (low + high) // 2
            shortened = {**document, "text": text[:middle]}
            if prompt_token_upper_bound(make_prompt(question, selected + [shortened], retry=True)) <= max_prompt_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            selected.append({**document, "text": text[:best]})
        break
    return selected


def parse_object(content: str) -> dict[str, Any] | None:
    """Принимает JSON даже если сервер по ошибке добавил короткое обрамление."""
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(content[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def normalize_answer(
    raw: dict[str, Any] | None,
    question: dict[str, Any],
    allowed_citations: set[str],
    allow_empty_citations: bool,
) -> dict[str, Any] | None:
    if not raw or not isinstance(raw.get("abstain"), bool):
        return None
    abstain = raw["abstain"]
    option_ids = {str(item["id"]) for item in question["options"]}
    answer_ids = raw.get("answer_ids")
    citations = raw.get("citations")
    if not isinstance(answer_ids, list) or not isinstance(citations, list):
        return None
    if not all(isinstance(item, str) for item in answer_ids + citations):
        return None
    if len(answer_ids) != len(set(answer_ids)) or len(citations) != len(set(citations)):
        return None
    if not set(answer_ids) <= option_ids or not set(citations) <= allowed_citations:
        return None
    rationale = safe_rationale(raw.get("rationale"), "Модель не дала обоснование.")
    if abstain:
        return {"answer_ids": [], "abstain": True, "rationale": rationale, "citations": []}
    if not answer_ids or len(citations) > 4 or (not citations and not allow_empty_citations):
        return None
    return {
        "answer_ids": answer_ids,
        "abstain": False,
        "rationale": rationale,
        "citations": citations,
    }


def usage_value(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None)
    return int(value) if isinstance(value, (int, float)) else None


def call_model(
    client: OpenAI,
    config: dict[str, Any],
    question_id: str,
    prompt: str,
) -> tuple[str, int, int]:
    model = config["model"]
    api_model = os.getenv("LLM_MODEL", model["name"])
    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=model["temperature"],
        top_p=model["top_p"],
        presence_penalty=model["presence_penalty"],
        max_tokens=model["max_tokens"],
        extra_headers={
            "X-Question-ID": question_id,
            "X-Submission-ID": os.getenv(model["submission_id_env"], "local-development"),
        },
        extra_body={
            "top_k": model["top_k"],
            "chat_template_kwargs": {"enable_thinking": model["enable_thinking"]},
        },
    )
    prompt_tokens = usage_value(response.usage, "prompt_tokens")
    completion_tokens = usage_value(response.usage, "completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError("Endpoint не вернул usage: расход нельзя проверить")
    content = response.choices[0].message.content or ""
    return content, prompt_tokens, completion_tokens


def answer_question(
    client: OpenAI,
    index: dict[str, Any],
    config: dict[str, Any],
    question: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    include_options = mode not in {"single-rag", "no-options", "no-context"}
    query = question_text(question, include_options=include_options)
    context, retrieved = select_context(index, query, mode)
    context = fit_context_to_limits(question, context, config)
    allowed = {str(item["block_id"]) for item in context}
    start = time.perf_counter()
    input_tokens = output_tokens = calls = 0
    answer: dict[str, Any] | None = None
    for retry in range(2):
        prompt = make_prompt(question, context, retry=bool(retry))
        # Оценка является верхней границей и предотвращает запрос сверх лимита.
        estimated_input = prompt_token_upper_bound(prompt)
        limits = config["limits"]
        if (
            calls >= limits["max_llm_calls"]
            or estimated_input + config["model"]["max_tokens"] > limits["max_single_request_tokens"]
            or input_tokens + estimated_input > limits["max_input_tokens"]
        ):
            break
        content, request_input, request_output = call_model(client, config, question["id"], prompt)
        calls += 1
        input_tokens += request_input
        output_tokens += request_output
        if (
            request_input + config["model"]["max_tokens"] > limits["max_single_request_tokens"]
            or input_tokens > limits["max_input_tokens"]
            or output_tokens > limits["max_output_tokens"]
        ):
            raise RuntimeError(f"Вопрос {question['id']}: превышен лимит токенов")
        answer = normalize_answer(
            parse_object(content), question, allowed, allow_empty_citations=(mode == "no-context")
        )
        if answer is not None:
            break
    if answer is None:
        answer = {
            "answer_ids": [],
            "abstain": True,
            "rationale": "Не удалось получить проверяемый ответ в требуемом формате.",
            "citations": [],
        }
    answer = {"id": question["id"], **answer}
    trace = {
        "id": question["id"],
        "llm_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "retrieved_block_ids": retrieved,
        "mode": mode,
    }
    return answer, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["final", "single-rag", "no-context", "no-char", "no-options"],
        default="final",
    )
    args = parser.parse_args()
    config = load_json(args.config)
    revision = config["model"].get("revision", "")
    if not isinstance(revision, str) or revision.startswith("PIN_") or len(revision) != 40:
        raise RuntimeError(
            "В config.json должна быть зафиксирована 40-символьная ревизия модели; "
            "сначала выполните pin_model_revision.py."
        )
    index = joblib.load(Path(args.index) / "index.joblib")
    questions = read_jsonl(args.questions)
    validate_questions(questions)
    base_url = os.getenv(config["model"]["base_url_env"])
    if not base_url:
        raise RuntimeError(f"Не задана переменная {config['model']['base_url_env']}")
    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv(config["model"]["api_key_env"], "EMPTY"),
        timeout=config["model"]["timeout_seconds"],
    )
    answers, traces = [], []
    for question in questions:
        answer, trace = answer_question(client, index, config, question, args.mode)
        answers.append(answer)
        traces.append(trace)
    write_jsonl(args.output, answers)
    write_jsonl(args.trace, traces)


if __name__ == "__main__":
    main()
