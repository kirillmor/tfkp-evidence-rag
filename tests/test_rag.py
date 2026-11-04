from __future__ import annotations

import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from tempfile import TemporaryDirectory

import fitz
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.evaluate import evaluate
from tfkp_rag.pipeline import answer_question, call_model, normalize_answer, select_context
from scripts.prepare_corpus import extract_corpus
from tfkp_rag.core import tokenize


def miniature_index() -> dict:
    documents = [
        {"block_id": "p_0001", "page": 1, "section": "§ 1", "text": "Условия Коши Римана для голоморфности."},
        {"block_id": "p_0002", "page": 2, "section": "§ 2", "text": "Теорема Морера и интеграл по треугольнику."},
        {"block_id": "p_0003", "page": 3, "section": "§ 3", "text": "Ряд Тейлора и коэффициенты производных."},
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    return {
        "documents": documents,
        "bm25": BM25Okapi([tokenize(row["text"]) for row in documents]),
        "vectorizer": vectorizer,
        "char_matrix": vectorizer.fit_transform([row["text"] for row in documents]),
        "config": {"retrieval": {"top_pages": 1, "neighbor_radius": 1, "max_context_chars": 10000, "bm25_weight": 0.65, "char_tfidf_weight": 0.35}},
    }


class RagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {"id": "q", "question": "Что говорит Морер?", "options": [{"id": "A", "text": "Верно"}, {"id": "B", "text": "Неверно"}, {"id": "C", "text": "Нет"}, {"id": "D", "text": "Да"}]}

    def test_context_contains_neighbors_and_source_ids(self) -> None:
        context, retrieved = select_context(miniature_index(), "Морера треугольник", "final")
        self.assertIn("p_0002", retrieved)
        self.assertEqual({row["block_id"] for row in context}, {"p_0001", "p_0002", "p_0003"})

    def test_answer_validation_rejects_foreign_citation(self) -> None:
        raw = {"answer_ids": ["A"], "abstain": False, "rationale": "Есть основания.", "citations": ["p_0099"]}
        self.assertIsNone(normalize_answer(raw, self.question, {"p_0002"}, False))

    def test_pdf_extraction_uses_stable_page_ids(self) -> None:
        with TemporaryDirectory() as directory:
            path = f"{directory}/book.pdf"
            document = fitz.open()
            document.new_page().insert_text((72, 72), "§ 2. Теорема\nТекст страницы")
            document.new_page().insert_text((72, 72), "Продолжение теоремы")
            document.save(path)
            document.close()
            rows = extract_corpus(path)
        self.assertEqual([row["block_id"] for row in rows], ["p_0001", "p_0002"])
        self.assertEqual(rows[1]["section"], "§ 2")

    def test_retry_is_logged(self) -> None:
        config = {"model": {"max_tokens": 700}, "limits": {"max_llm_calls": 2, "max_input_tokens": 16384, "max_output_tokens": 4096, "max_single_request_tokens": 8192}}
        responses = [("не JSON", 20, 5), ('{"answer_ids":["A"],"abstain":false,"rationale":"Подтверждено.","citations":["p_0002"]}', 20, 5)]
        with patch("tfkp_rag.pipeline.call_model", side_effect=responses):
            answer, trace = answer_question(None, miniature_index(), config, self.question, "final")
        self.assertEqual(answer["answer_ids"], ["A"])
        self.assertEqual(trace["llm_calls"], 2)
        self.assertEqual(trace["input_tokens"], 40)

    def test_openai_compatible_request_has_required_headers(self) -> None:
        received: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - имя задаёт стандартная библиотека.
                received["question"] = self.headers.get("X-Question-ID", "")
                received["submission"] = self.headers.get("X-Submission-ID", "")
                payload = {
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
                body = __import__("json").dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Песочница запрещает открыть локальный тестовый порт")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="EMPTY")
            config = {"model": {"name": "test-model", "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 20, "enable_thinking": False, "submission_id_env": "TEST_SUBMISSION"}}
            with patch.dict("os.environ", {"TEST_SUBMISSION": "submission-42"}):
                _, prompt_tokens, completion_tokens = call_model(client, config, "question-7", "Проверка")
            self.assertEqual((prompt_tokens, completion_tokens), (11, 7))
            self.assertEqual(received, {"question": "question-7", "submission": "submission-42"})
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluator_counts_citation_f1(self) -> None:
        question = {"id": "q", "question": "x", "options": self.question["options"]}
        answer = {"id": "q", "answer_ids": ["A"], "abstain": False, "rationale": "x", "citations": ["p_0001"]}
        gold = {"id": "q", "category": "conditions", "answer_ids": ["A"], "abstain": False, "acceptable_citation_sets": [["p_0001"]]}
        metrics = evaluate([question], [answer], [gold])
        self.assertEqual(metrics["Q"], 1.0)


if __name__ == "__main__":
    unittest.main()
