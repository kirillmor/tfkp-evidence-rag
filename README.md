# Доказательный RAG по ТФКП

**Автор:** `ФИО`  
**Группа:** `ГРУППА`

Система отвечает на тестовые вопросы только по учебнику В. В. Горяйнова и
Е. С. Половинкина «Лекции по ТФКП» (2025), приводит короткое обоснование и
цитирует исходные постраничные блоки `p_XXXX`.

## Требования

- Python 3.11;
- локально запущенная Qwen/Qwen3.5-4B с OpenAI-совместимым API;
- сеть нужна только один раз для скачивания PDF и весов модели.

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

## Подготовка источника и индекса

```bash
python download_pdf.py --output data/raw/tfkp_2025.pdf
python prepare_corpus.py --pdf data/raw/tfkp_2025.pdf --output data/generated/corpus.jsonl
python prepare_datasets.py
python pin_model_revision.py --config config.json
python build_index.py --corpus data/generated/corpus.jsonl --output ./index --config config.json
```

`download_pdf.py` сохраняет SHA-256 PDF рядом с файлом. `pin_model_revision.py`
нужно выполнить один раз перед сдачей: он заменяет плейсхолдер в `config.json`
на неизменяемый commit SHA модели.

Корпус содержит один блок на физическую страницу PDF. Поля корпуса:
`block_id`, `page`, `section`, `text`; ID стабилен и равен `p_0001`, `p_0002`
и так далее.

## Локальная модель и предсказание

Пример запуска vLLM (в его отдельном окружении):

```bash
vllm serve Qwen/Qwen3.5-4B --revision <SHA_из_config.json> --host 127.0.0.1 --port 8000
```

Затем в терминале решения:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=EMPTY
export SUBMISSION_ID=local-dev
python predict.py --index ./index --questions dev_questions.jsonl \
  --output dev_answers.jsonl --trace dev_trace.jsonl --config config.json
python evaluate.py --questions dev_questions.jsonl --answers dev_answers.jsonl \
  --gold dev_gold.jsonl --trace dev_trace.jsonl
```

`predict.py` посылает `X-Question-ID` и `X-Submission-ID` в каждом запросе.
Он использует максимум два вызова на вопрос: повтор возможен только при
некорректном JSON. При недоступном или не возвращающем usage endpoint скрипт
завершается ошибкой, а не записывает непроверяемый расход.

## Архитектура

Индекс объединяет BM25 по нормализованным словам и символьный TF-IDF по
3–5-граммам. В итоговом режиме запрос включает вопрос и варианты; три лучших
страницы расширяются соседними страницами, а контекст ограничен 18 000
символами. В prompt разрешены цитаты только из переданного контекста.

В репозиторий не добавляются PDF, корпус, индекс, веса, кэши, ключи и большие
логи. `dev_answers.jsonl`, `dev_trace.jsonl` и заполненный `experiments.csv`
нужно добавить после фактического прогона перед сдачей.

## Эксперименты и проверки

```bash
python run_experiments.py --index ./index
python -m unittest discover -s tests -v
```

Скрипт экспериментов запускает baseline без контекста, простой BM25 RAG,
итоговую конфигурацию и две абляции. Метрики не зашиваются в код: CSV
формируется из фактических ответов и trace.
