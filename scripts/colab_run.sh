#!/usr/bin/env bash
# Полный запуск задания в одной GPU-сессии Google Colab.
# Запуск из корня клонированного репозитория: !bash scripts/colab_run.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PDF_PATH="${PDF_PATH:-data/raw/tfkp_2025.pdf}"
INDEX_DIR="${INDEX_DIR:-index}"
RESULTS_DIR="${COLAB_RESULTS_DIR:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
VLLM_PID=""

cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

on_error() {
  local exit_code=$?
  echo "Пайплайн остановлен с кодом ${exit_code}."
  if [[ -f logs/vllm.log ]]; then
    echo "Последние строки logs/vllm.log:"
    tail -n 80 logs/vllm.log || true
  fi
  exit "$exit_code"
}

trap cleanup EXIT
trap on_error ERR

echo "Проверка GPU..."
nvidia-smi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
  echo "Требуется Python 3.11, обнаружен Python ${PYTHON_VERSION}."
  echo "Выберите в Colab runtime с Python 3.11 и повторите запуск."
  exit 1
fi

echo "Установка зависимостей проекта и vLLM..."
$PYTHON_BIN -m pip install --quiet -r requirements.txt
$PYTHON_BIN -m pip install --quiet vllm

if [[ ! -f "$PDF_PATH" ]]; then
  echo "Скачивание PDF учебника..."
  $PYTHON_BIN -m scripts.download_pdf --output "$PDF_PATH" || {
    echo "Не удалось скачать PDF. Загрузите именно учебник из условия в Colab"
    echo "и повторите запуск с PDF_PATH=/путь/к/учебнику.pdf."
    exit 1
  }
fi

echo "Подготовка корпуса, наборов данных и индекса..."
$PYTHON_BIN -m scripts.prepare_corpus --pdf "$PDF_PATH" --output data/generated/corpus.jsonl
$PYTHON_BIN -m scripts.prepare_datasets
$PYTHON_BIN build_index.py --corpus data/generated/corpus.jsonl --output "$INDEX_DIR" --config config.json

MODEL_REVISION="$($PYTHON_BIN -c 'import json; print(json.load(open("config.json", encoding="utf-8"))["model"]["revision"])')"
if [[ "$MODEL_REVISION" == PIN_* ]] || [[ ${#MODEL_REVISION} -ne 40 ]]; then
  echo "Фиксация ревизии Qwen..."
  $PYTHON_BIN -m scripts.pin_model_revision --config config.json
  MODEL_REVISION="$($PYTHON_BIN -c 'import json; print(json.load(open("config.json", encoding="utf-8"))["model"]["revision"])')"
fi

mkdir -p logs
echo "Запуск vLLM с ревизией ${MODEL_REVISION}..."
vllm serve Qwen/Qwen3.5-4B \
  --revision "$MODEL_REVISION" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  >logs/vllm.log 2>&1 &
VLLM_PID=$!

echo "Ожидание готовности сервера модели..."
for ((attempt = 1; attempt <= VLLM_START_TIMEOUT / 2; attempt++)); do
  if curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM завершился до готовности."
    exit 1
  fi
  sleep 2
done

if ! curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "vLLM не стал доступен за ${VLLM_START_TIMEOUT} секунд."
  exit 1
fi

export LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SUBMISSION_ID="${SUBMISSION_ID:-colab-dev}"

echo "Итоговый прогон на открытом наборе..."
$PYTHON_BIN predict.py \
  --index "$INDEX_DIR" \
  --questions dev_questions.jsonl \
  --output dev_answers.jsonl \
  --trace dev_trace.jsonl \
  --config config.json

echo "Проверка итогового прогона..."
$PYTHON_BIN -m scripts.evaluate \
  --questions dev_questions.jsonl \
  --answers dev_answers.jsonl \
  --gold dev_gold.jsonl \
  --trace dev_trace.jsonl

echo "Запуск baseline, итоговой системы и абляций..."
$PYTHON_BIN -m scripts.run_experiments --index "$INDEX_DIR"

if [[ -n "$RESULTS_DIR" ]]; then
  mkdir -p "$RESULTS_DIR"
  cp dev_answers.jsonl dev_trace.jsonl experiments.csv "$RESULTS_DIR/"
  echo "Результаты скопированы в ${RESULTS_DIR}."
fi

echo "Пайплайн завершён. Добавьте dev_answers.jsonl, dev_trace.jsonl и experiments.csv в Git."
