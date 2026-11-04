#!/usr/bin/env python3
"""Проверяет формат ответов и считает метрики, определённые в задании."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any

from tfkp_rag.core import read_jsonl, validate_questions


def citation_f1(predicted: list[str], acceptable: list[list[str]]) -> float:
    if not predicted or not acceptable:
        return 0.0
    predicted_set = set(predicted)
    result = 0.0
    for target in acceptable:
        target_set = set(target)
        intersection = len(predicted_set & target_set)
        if intersection:
            result = max(result, 2 * intersection / (len(predicted_set) + len(target_set)))
    return result


def validate_answers(
    questions: list[dict[str, Any]], answers: list[dict[str, Any]], allow_empty_citations: bool
) -> dict[str, dict[str, Any]]:
    question_map = {row["id"]: row for row in questions}
    if len(answers) != len(questions):
        raise ValueError("Число ответов не совпадает с числом вопросов")
    result: dict[str, dict[str, Any]] = {}
    for row in answers:
        question_id = row.get("id")
        if question_id not in question_map or question_id in result:
            raise ValueError("В выходе есть неизвестный или повторяющийся ID")
        if not isinstance(row.get("abstain"), bool):
            raise ValueError(f"{question_id}: abstain должен быть boolean")
        answer_ids, citations = row.get("answer_ids"), row.get("citations")
        if not isinstance(answer_ids, list) or not isinstance(citations, list):
            raise ValueError(f"{question_id}: answer_ids и citations должны быть списками")
        if len(answer_ids) != len(set(answer_ids)) or len(citations) != len(set(citations)):
            raise ValueError(f"{question_id}: списки не должны иметь повторов")
        possible = {option["id"] for option in question_map[question_id]["options"]}
        if not set(answer_ids) <= possible or len(citations) > 4:
            raise ValueError(f"{question_id}: недопустимый вариант или слишком много ссылок")
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1200:
            raise ValueError(f"{question_id}: некорректное rationale")
        if row["abstain"] and (answer_ids or citations):
            raise ValueError(f"{question_id}: при отказе списки должны быть пустыми")
        if not row["abstain"] and (not answer_ids or (not citations and not allow_empty_citations)):
            raise ValueError(f"{question_id}: разрешимый ответ требует вариантов и ссылок")
        result[question_id] = row
    return result


def evaluate(
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    traces: list[dict[str, Any]] | None = None,
    allow_empty_citations: bool = False,
) -> dict[str, Any]:
    validate_questions(questions)
    answer_map = validate_answers(questions, answers, allow_empty_citations)
    gold_map = {row["id"]: row for row in gold}
    if set(gold_map) != set(answer_map):
        raise ValueError("ID эталонов и ответов не совпадают")
    category_scores: dict[str, list[float]] = defaultdict(list)
    exact: list[float] = []
    total_scores: list[float] = []
    for question_id, prediction in answer_map.items():
        target = gold_map[question_id]
        correct = (
            bool(prediction["abstain"]) == bool(target["abstain"])
            and set(prediction["answer_ids"]) == set(target["answer_ids"])
        )
        exact.append(float(correct))
        if target["abstain"]:
            score = float(correct)
        else:
            f1 = citation_f1(prediction["citations"], target["acceptable_citation_sets"])
            score = (0.8 + 0.2 * f1) if correct else 0.0
        category_scores[target["category"]].append(score)
        total_scores.append(score)
    category_means = {name: sum(values) / len(values) for name, values in category_scores.items()}
    metrics: dict[str, Any] = {
        "answer_accuracy": sum(exact) / len(exact),
        "mean_score": sum(total_scores) / len(total_scores),
        "category_means": category_means,
        "Q": sum(category_means.values()) / len(category_means),
    }
    if traces is not None:
        trace_map = {row.get("id"): row for row in traces}
        if set(trace_map) != set(answer_map):
            raise ValueError("ID trace и ответов не совпадают")
        costs, inputs, outputs, calls, latency = [], [], [], [], []
        for question_id in answer_map:
            trace = trace_map[question_id]
            for field in ("llm_calls", "input_tokens", "output_tokens", "elapsed_ms"):
                if not isinstance(trace.get(field), (int, float)) or trace[field] < 0:
                    raise ValueError(f"{question_id}: некорректное поле trace {field}")
            inputs.append(trace["input_tokens"])
            outputs.append(trace["output_tokens"])
            calls.append(trace["llm_calls"])
            latency.append(trace["elapsed_ms"])
            costs.append(trace["input_tokens"] + 4 * trace["output_tokens"] + 256 * trace["llm_calls"])
        metrics.update(
            {
                "mean_input_tokens": sum(inputs) / len(inputs),
                "mean_output_tokens": sum(outputs) / len(outputs),
                "mean_llm_calls": sum(calls) / len(calls),
                "mean_cost": sum(costs) / len(costs),
                "median_latency_ms": statistics.median(latency),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--allow-empty-citations", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(
        read_jsonl(args.questions),
        read_jsonl(args.answers),
        read_jsonl(args.gold),
        read_jsonl(args.trace) if args.trace else None,
        args.allow_empty_citations,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
