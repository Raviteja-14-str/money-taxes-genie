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
import time
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
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_ITEM_CHARS = 1200
MIN_RELEVANCE_SCORE = 2.0


SYSTEM_PROMPT = """You are Money & Taxes Genie, a plain-language assistant that explains Indian personal finance and tax concepts.
Rules:
- Answer ONLY using the CONTEXT provided below.
- If the context doesn't cover the question, say so plainly instead of guessing.
- Earlier user questions are only for resolving references such as "it" or "that"; never treat an earlier assistant answer as a factual source.
- This is general education, never personalized financial or tax advice — don't tell the user what they specifically should do with their money.
- Keep answers clear and to the point.
- Never ask for or repeat PAN, Aadhaar, bank-account, card, password, or other secret information.
"""


ABOUT_RESPONSE = (
    "I’m Money & Taxes Genie. I retrieve relevant, source-attributed Indian finance and tax "
    "learning documents, then use a local or cloud language model to explain them in plain language. "
    "I do not access bank accounts, perform transactions, file returns, or provide personalised "
    "financial, tax, or investment advice."
)

SAFETY_RESPONSE = (
    "I can explain the process and provide a general checklist, but I cannot file, sign in, submit, pay, "
    "e-verify, or use an OTP for you. Please keep OTPs, passwords, PAN/Aadhaar numbers, and account or card "
    "details private."
)

NO_CONTEXT_RESPONSE = (
    "I don’t have a reliable source for that in my current knowledge files. "
    "Please ask about an Indian money or tax concept covered by the assistant, or add an authoritative source first."
)


STOPWORDS = {
    "a", "about", "after", "also", "am", "an", "and", "are", "as", "at", "be",
    "can", "could", "do", "does", "explain", "for", "from", "have", "help", "how",
    "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "please", "tell",
    "that", "the", "this", "to", "what", "when", "where", "which", "with", "would",
    "you", "your", "india", "indian", "today", "now", "current", "latest",
}


def words(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", value)
        if token.lower() not in STOPWORDS
    }


def contains_term(text: str, term: str) -> bool:
    """Match abbreviations as words and phrases as normalized substrings."""
    if len(term) <= 3:
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return term in text


def normalize_question(value: str) -> str:
    """Normalize common finance abbreviations, spelling variants, and typos for retrieval."""
    normalized = re.sub(r"\s+", " ", value.lower()).strip()
    replacements = (
        (r"\bpay\s*slip\b|\bpayslip\b|\bsalary statement\b", "salary slip"),
        (r"\bprovisional\s+funds?\b", "provident fund"),
        (r"\bprovident\s+funds?\b", "provident fund"),
        (r"\bpf\b|\bepf\b", "provident fund"),
        (r"\bform\s*26\s*as\b|\b26\s+as\b", "26as"),
        (r"\bform\s*16\b", "form16"),
        (r"\btax\s+deducted\s+at\s+source\b", "tds"),
        (r"\bmutual\s+funds?\b", "mutual fund"),
        (r"\bupi\s+pin\b", "upi pin"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def sanitize_history(history: Any) -> list[str]:
    """Keep only short earlier user questions; assistant text is not trusted as source material."""
    if not isinstance(history, list):
        return []
    questions = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            questions.append(content[:MAX_HISTORY_ITEM_CHARS])
    return questions[-MAX_HISTORY_MESSAGES:]


def is_about_question(question: str) -> bool:
    normalized = normalize_question(question)
    return bool(re.search(r"\b(how do you work|what can you do|who are you|what is money genie)\b", normalized))


def is_protected_action_question(question: str) -> bool:
    normalized = normalize_question(question)
    return bool(
        re.search(r"\b(file|submit|pay|e[- ]?verify)\b.*\b(return|tax|bill|transaction)\b", normalized)
        or re.search(r"\b(use|enter|share|send|provide)\b.*\b(my\s+)?otp\b", normalized)
        or re.search(r"\b(sign in|log in|login)\b.*\b(my\s+)?(bank|account)\b", normalized)
    )


def is_follow_up(question: str) -> bool:
    normalized = normalize_question(question)
    return bool(re.search(r"\b(it|that|this|they|them|same|there|then)\b|\bwhat about\b", normalized))


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


def retrieve(question: str, top_k: int = 4) -> list[tuple[float, dict[str, Any]]]:
    lowered_question = normalize_question(question)
    question_words = words(lowered_question)
    focused_source = None
    original_question = question.lower()
    if re.search(r"\b(nps|ppf|retirement|pension)\b", original_question):
        focused_source = "retirement-pension-savings.txt"
    elif re.search(r"\b(first[- ]time investor|new investor)\b", original_question):
        focused_source = "mutual-funds-sip-basics.txt"
    elif re.search(r"\b(pf|epf|uan|eps|edli|provisional funds?)\b", original_question) or "provident fund" in lowered_question:
        focused_source = "provident-fund-pf.txt"
    elif re.search(r"\b(form\s*16|form16)\b", original_question):
        focused_source = "form16-basics.txt"
    elif re.search(r"\b(ais|26as|tds)\b", original_question):
        focused_source = "income-tax-compliance-workflow.txt"
    elif re.search(r"\b(nps|ppf)\b", original_question):
        focused_source = "retirement-pension-savings.txt"
    elif re.search(r"\b(upi)\b", original_question):
        focused_source = "digital-payments-safety.txt"
    elif re.search(r"\b(gst|cgst|sgst|igst)\b", original_question):
        focused_source = "gst-basics.txt"
    topic_hints = (
        (("provident fund", "pf", "epf", "uan", "eps", "edli"), "provident-fund-pf.txt"),
        (("fixed pay", "variable pay", "ctc", "gross pay", "take home", "net pay", "bonus", "incentive"), "salary-pay-components.txt"),
        (("salary slip", "basic pay", "hra"), "understanding-salary-slip.txt"),
        (("form16",), "form16-basics.txt"),
        (("fy", "ay", "assessment year", "financial year", "tax year", "tds", "ais", "26as"), "income-tax-compliance-workflow.txt"),
        (("old regime", "new regime", "tax regime", "115bac"), "old-vs-new-tax-regime.txt"),
        (("deduction", "rebate", "exemption", "80c", "80d"), "tax-deductions-and-exemptions.txt"),
        (("budget", "cash flow", "emergency fund", "saving", "savings"), "budgeting-cash-flow.txt"),
        (("bank account", "fixed deposit", "recurring deposit", "kyc", "nomination", "dicgc"), "bank-accounts-deposits-kyc.txt"),
        (("loan", "emi", "interest rate", "credit score", "credit card", "borrowing"), "borrowing-credit-loans.txt"),
        (("upi", "digital payment", "qr code", "unauthorised transaction"), "digital-payments-safety.txt"),
        (("first-time investor", "new investor"), "mutual-funds-sip-basics.txt"),
        (("sip", "nav", "direct plan", "regular plan", "expense ratio", "exit load", "mutual fund"), "mutual-funds-and-sip-expanded.txt"),
        (("diversification", "shares", "stocks", "bonds", "demat", "investment risk"), "investing-risk-and-markets.txt"),
        (("health insurance", "life insurance", "claim", "co-pay", "waiting period", "sum insured", "policy"), "insurance-policy-and-claims.txt"),
        (("nps", "ppf", "retirement", "pension"), "retirement-pension-savings.txt"),
        (("scam", "fraud", "unauthorised", "complaint"), "financial-scams-and-complaints.txt"),
        (("gst", "cgst", "sgst", "igst"), "gst-basics.txt"),
    )
    scored = []
    for document in load_documents():
        if focused_source and document["source"] != focused_source:
            continue
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
                phrase_bonus += 8.0 if document["source"] == preferred_source else 0.0
        score = overlap + phrase_bonus
        if score >= MIN_RELEVANCE_SCORE:
            scored.append((score, document))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["source"]))
    return scored[:top_k]


def build_retrieval_question(question: str, history: list[str]) -> str:
    if history and is_follow_up(question):
        return f"{history[-1]} {question}"
    return question


def build_messages(question: str, context: str, history: list[str] | None = None) -> list[dict[str, str]]:
    earlier = ""
    if history:
        earlier = "EARLIER USER QUESTIONS (reference only):\n" + "\n".join(f"- {item}" for item in history) + "\n\n"
    user_message = f"{earlier}KNOWLEDGE CONTEXT:\n{context}\n\nCURRENT QUESTION: {question}"
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
    last_error = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 0:
                time.sleep(0.5)
    else:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start Ollama and run '{OLLAMA_MODEL}'."
        ) from last_error
    return clean_model_output(result["message"]["content"])


def call_huggingface_local(messages: list[dict[str, str]]) -> str:
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face local dependencies are missing. Install local/requirements-hf.txt."
        ) from error

    generator = get_hf_pipeline()
    prompt: Any = messages
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Some Transformers/tokenizer combinations do not expose Qwen's
            # enable_thinking switch. The output cleaner remains a fallback.
            prompt = messages
    result = generator(
        prompt,
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


def source_payload(matches: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_title": document["title"],
            "relevance": round(score, 3),
            "source_urls": document.get("source_urls", []),
        }
        for score, document in matches
    ]


def direct_answer(question: str, matches: list[tuple[float, dict[str, Any]]]) -> str | None:
    lowered = question.lower()
    if re.search(r"\b(pf|epf|provisional funds?)\b", lowered) and re.search(r"\b(what|meaning|stand|is|pay slip|payslip)\b", lowered):
        return (
            "PF usually means Provident Fund, specifically EPF (Employees’ Provident Fund) in an Indian payslip. "
            "It is a retirement-savings contribution; the employee contribution is deducted from pay, while the employer contribution is shown separately according to the applicable rules."
        )
    if "provisional" in lowered and "fund" in lowered:
        return "The usual term is Provident Fund, not provisional funds. In an Indian payslip, PF generally refers to Employees’ Provident Fund (EPF)."
    return None


def answer_question(question: str, history: Any = None) -> tuple[str, list[dict[str, Any]]]:
    history_questions = sanitize_history(history)
    if is_protected_action_question(question):
        return SAFETY_RESPONSE, []
    if is_about_question(question):
        return ABOUT_RESPONSE, []

    retrieval_question = build_retrieval_question(question, history_questions)
    matches = retrieve(retrieval_question)
    if not matches:
        return NO_CONTEXT_RESPONSE, []

    best_score = matches[0][0]
    if best_score >= 8.0:
        matches = [match for match in matches if match[0] >= best_score * 0.4]

    direct = direct_answer(question, matches)
    if direct:
        return direct, source_payload(matches)

    if matches:
        context = "\n\n---\n\n".join(
            f"[Source: {document['title']}]\n{document['text']}"
            for _score, document in matches
        )
    else:
        return NO_CONTEXT_RESPONSE, []

    messages = build_messages(question, context, history_questions)
    if PROVIDER == "ollama":
        answer = call_ollama(messages)
    elif PROVIDER == "hf_local":
        answer = call_huggingface_local(messages)
    elif PROVIDER == "hf_api":
        answer = call_huggingface_api(messages)
    else:
        raise RuntimeError("MG_PROVIDER must be ollama, hf_local, or hf_api.")

    return answer, source_payload(matches)


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
            answer, sources = answer_question(question, body.get("history"))
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
