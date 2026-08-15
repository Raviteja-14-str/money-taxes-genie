"""
Money & Taxes Genie — local learning server
---------------------------------------------

This server keeps the browser contract the same as the Bedrock Lambda:
POST /api/chat with {"question": "..."} and receive an answer plus sources.

The retrieval step is deliberately small and readable: it ranks the local
knowledge files by keyword overlap. The answer model is swappable through
MG_PROVIDER:

  ollama   - local Ollama server at http://127.0.0.1:11434
  hf_local - Hugging Face Transformers loaded into this Python process
  hf_api   - Hugging Face hosted inference, using HF_TOKEN from the environment

No API key is accepted from the browser and no secret is written to disk.
"""

from __future__ import annotations

import html
import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
KNOWLEDGE_ROOT = Path(os.environ.get("KNOWLEDGE_ROOT", str(ROOT)))
HOST = os.environ.get("LOCAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_PORT", "3000"))
PROVIDER = os.environ.get("MG_PROVIDER", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3-0.6B")
HF_PROVIDER = os.environ.get("HF_PROVIDER")
MAX_QUESTION_CHARS = 4000
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "700"))


SYSTEM_PROMPT = """You are Money & Taxes Genie, a plain-language assistant that explains Indian personal finance and tax concepts.
Rules:
- Answer ONLY using the CONTEXT provided below.
- If the context doesn't cover the question, say so plainly instead of guessing.
- This is general education, never personalized financial or tax advice — don't tell the user what they specifically should do with their money.
- Keep answers clear and to the point.
- Never ask for or repeat PAN, Aadhaar, bank-account, card, password, or other secret information.
"""


STOPWORDS = {
    "about", "after", "also", "and", "are", "can", "could", "does", "explain",
    "from", "have", "help", "how", "into", "what", "when", "where", "which",
    "with", "would", "your", "the", "this", "that", "for", "india", "indian",
}


def words(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", value)
        if token.lower() not in STOPWORDS
    }


def contains_term(text: str, term: str) -> bool:
    """Match short tax abbreviations as words, not as substrings of other words."""
    if len(term) <= 2:
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return term in text


def load_documents() -> list[dict[str, Any]]:
    documents = []
    for path in sorted(KNOWLEDGE_ROOT.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.name
        first_line = text.splitlines()[0].strip()
        if first_line.upper().startswith("TITLE:"):
            title = first_line.split(":", 1)[1].strip()
        source_urls = [url.rstrip(".,)") for url in re.findall(r"https?://[^\s]+", text)]
        documents.append({
            "title": title,
            "source": path.name,
            "source_urls": source_urls,
            "text": text,
        })
    return documents


def retrieve(question: str, top_k: int = 4) -> list[tuple[float, dict[str, str]]]:
    question_words = words(question)
    lowered_question = question.lower()
    topic_hints = (
        (("fy", "ay", "assessment year", "financial year", "tax year"), "income-tax-compliance-workflow.txt"),
        (("ais", "26as", "26 as", "annual information statement"), "income-tax-compliance-workflow.txt"),
        (("deduction", "rebate", "exemption"), "tax-deductions-and-exemptions.txt"),
    )
    scored = []
    for document in load_documents():
        document_words = words(document["text"])
        overlap = len(question_words & document_words)
        phrase_bonus = 0.0
        lowered_text = document["text"].lower()
        for phrase in (
            "tax regime", "salary slip", "fixed pay", "variable pay", "mutual fund",
            "provident fund", "form 16", "bank account", "emergency fund", "cash flow",
            "credit score", "digital payment", "upi", "insurance", "retirement", "nps",
            "ppf", "gst", "ais", "26as", "tds",
        ):
            if phrase in lowered_question and phrase in lowered_text:
                phrase_bonus += 1.5
        for terms, preferred_source in topic_hints:
            if any(contains_term(lowered_question, term) for term in terms):
                phrase_bonus += 5.0 if document["source"] == preferred_source else 0.0
        score = overlap + phrase_bonus
        if score > 0:
            scored.append((score, document))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def clean_model_output(value: str) -> str:
    """Hide Qwen-style internal reasoning markers from the browser response."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def call_ollama(messages: list[dict[str, str]]) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }).encode("utf-8")
    request = Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start Ollama and run '{OLLAMA_MODEL}'."
        ) from error
    return clean_model_output(result["message"]["content"])


def call_huggingface_local(messages: list[dict[str, str]]) -> str:
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face local dependencies are missing. Install local/requirements-hf.txt."
        ) from error

    generator = get_hf_pipeline()
    result = generator(
        messages,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    generated = result[0].get("generated_text", "")
    if isinstance(generated, list):
        for message in reversed(generated):
            if message.get("role") == "assistant":
                return clean_model_output(message.get("content", ""))
        return ""
    return clean_model_output(str(generated))


def call_huggingface_api(messages: list[dict[str, str]]) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the hf_api provider and must stay in the environment.")
    try:
        from huggingface_hub import InferenceClient
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face hosted dependencies are missing. Install local/requirements-hf.txt."
        ) from error

    client = InferenceClient(provider=HF_PROVIDER, token=token)
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )
    return clean_model_output(response.choices[0].message.content)


_HF_PIPELINE: Any = None


def get_hf_pipeline() -> Any:
    global _HF_PIPELINE
    if _HF_PIPELINE is None:
        from transformers import pipeline

        _HF_PIPELINE = pipeline(
            "text-generation",
            model=HF_MODEL,
            device_map="auto",
            torch_dtype="auto",
        )
    return _HF_PIPELINE


def answer_question(question: str) -> tuple[str, list[dict[str, Any]]]:
    matches = retrieve(question)
    if matches:
        context = "\n\n---\n\n".join(
            f"[Source: {document['title']}]\n{document['text']}"
            for _score, document in matches
        )
    else:
        context = "(no relevant sources found in the local knowledge files)"

    messages = build_messages(question, context)
    if PROVIDER == "ollama":
        answer = call_ollama(messages)
    elif PROVIDER == "hf_local":
        answer = call_huggingface_local(messages)
    elif PROVIDER == "hf_api":
        answer = call_huggingface_api(messages)
    else:
        raise RuntimeError("MG_PROVIDER must be ollama, hf_local, or hf_api.")

    sources = [
        {
            "doc_title": document["title"],
            "relevance": round(score, 3),
            "source_urls": document.get("source_urls", []),
        }
        for score, document in matches
    ]
    return answer, sources


class Handler(BaseHTTPRequestHandler):
    server_version = "MoneyGenieLocal/1.0"

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"ok": True, "provider": PROVIDER})
            return

        requested = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        candidate = (WEB_ROOT / requested).resolve()
        if WEB_ROOT not in candidate.parents and candidate != WEB_ROOT:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(str(candidate))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/api/chat":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            question = (body.get("question") or "").strip()
            if not question:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "No question provided."})
                return
            if len(question) > MAX_QUESTION_CHARS:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Question is too long."})
                return
            answer, sources = answer_question(question)
            self.send_json(HTTPStatus.OK, {"answer": answer, "sources": sources})
        except Exception as error:  # noqa: BLE001 - public API must not leak internals
            print(f"local provider error: {error}", file=sys.stderr)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "Money & Taxes Genie is temporarily unavailable.",
            })

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    print(f"Money & Taxes Genie local server: http://{HOST}:{PORT}")
    print(f"Provider: {PROVIDER}")
    if PROVIDER == "ollama":
        print(f"Ollama model: {OLLAMA_MODEL}")
    elif PROVIDER.startswith("hf"):
        print(f"Hugging Face model: {HF_MODEL}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
