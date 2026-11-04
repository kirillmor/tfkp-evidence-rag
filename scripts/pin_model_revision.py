#!/usr/bin/env python3
"""Фиксирует текущий commit SHA модели Hugging Face в config.json перед запуском."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    path = Path(args.config)
    config = json.loads(path.read_text(encoding="utf-8"))
    model_name = config["model"]["name"]
    with urlopen(f"https://huggingface.co/api/models/{model_name}", timeout=30) as response:
        payload = json.load(response)
    revision = payload.get("sha")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("Hugging Face API не вернул commit SHA модели")
    config["model"]["revision"] = revision
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Зафиксирована ревизия {model_name}: {revision}")


if __name__ == "__main__":
    main()
