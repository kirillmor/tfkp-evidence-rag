#!/usr/bin/env python3
"""Запускает обязательные пять конфигураций и записывает честную таблицу CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from scripts.evaluate import evaluate
from tfkp_rag.core import read_jsonl


EXPERIMENTS = [
    ("no_context", "no-context", True),
    ("single_rag", "single-rag", False),
    ("final", "final", False),
    ("ablation_no_char_tfidf", "no-char", False),
    ("ablation_no_option_query", "no-options", False),
]
FIELDS = [
    "name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens",
    "mean_llm_calls", "mean_cost", "median_latency_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", default="dev_questions.jsonl")
    parser.add_argument("--gold", default="dev_gold.jsonl")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="experiments.csv")
    parser.add_argument("--runs-dir", default="experiments_runs")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    questions, gold = read_jsonl(args.questions), read_jsonl(args.gold)
    rows = []
    for name, mode, allow_empty_citations in EXPERIMENTS:
        answers = runs_dir / f"{name}_answers.jsonl"
        trace = runs_dir / f"{name}_trace.jsonl"
        command = [
            sys.executable, "predict.py", "--index", args.index, "--questions", args.questions,
            "--output", str(answers), "--trace", str(trace), "--config", args.config,
            "--mode", mode,
        ]
        subprocess.run(command, check=True)
        metrics = evaluate(
            questions, read_jsonl(answers), gold, read_jsonl(trace), allow_empty_citations
        )
        rows.append(
            {
                "name": name,
                "answer_accuracy": f"{metrics['answer_accuracy']:.4f}",
                "Q": "" if name == "no_context" else f"{metrics['Q']:.4f}",
                **{field: f"{metrics[field]:.2f}" for field in FIELDS[3:]},
            }
        )
    with Path(args.output).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Результаты записаны в {args.output}")


if __name__ == "__main__":
    main()
