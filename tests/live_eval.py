"""Synthetic end-to-end checks for a running local Money Genie server."""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("MONEY_GENIE_URL", "http://127.0.0.1:3000")

CASES = [
    ("What is PF in my payslip?", "Provident Fund", "What Is PF"),
    ("Isn't it provisional funds?", "Provident Fund", "What Is PF"),
    ("How do you work?", "retrieve relevant", None),
    ("What is Bitcoin's price today?", "reliable source", None),
    ("What is fixed pay versus variable pay?", "Fixed pay", "Fixed Pay"),
    ("What is CTC versus take-home pay?", "CTC", "Fixed Pay"),
    ("What is an emergency fund?", "emergency fund", "Budgeting"),
    ("Why can a longer loan tenure cost more?", "interest", "Loans"),
    ("Is a UPI PIN needed to receive money?", "not needed", "UPI"),
    ("What is the difference between FY and AY?", "financial year", "Income Tax Workflow"),
    ("What is AIS and Form 26AS?", "AIS", "Income Tax Workflow"),
    ("What is the difference between a tax deduction and a rebate?", "deduction", "Tax Deductions"),
    ("Why are mutual funds not guaranteed returns?", "risk", "Mutual Funds"),
    ("What should I check in health insurance?", "coverage", "Life and Health Insurance"),
    ("What is NPS?", "pension", "EPF"),
    ("What are signs of a financial scam?", "guaranteed", "Financial Scams"),
    ("What is GST?", "GST", "GST"),
    ("Can you file my tax return or use my OTP?", "cannot file", None),
]


def post(question: str, history=None) -> dict:
    body = json.dumps({"question": question, "history": history or []}).encode()
    request = Request(
        f"{BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last_error = None
    for _attempt in range(2):
        try:
            with urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode())
        except (HTTPError, OSError) as error:
            last_error = error
            time.sleep(0.5)
    raise last_error


def main() -> int:
    with urlopen(f"{BASE_URL}/health", timeout=10) as response:
        health = json.loads(response.read().decode())
    print("health:", health)

    failures = []
    for question, answer_marker, source_marker in CASES:
        try:
            result = post(question)
        except Exception as error:  # noqa: BLE001 - the evaluator should report all cases
            failures.append((question, "request failed", str(error)))
            print("FAIL", question, "->", error)
            continue
        answer = result.get("answer", "")
        source_text = " ".join(source.get("doc_title", "") for source in result.get("sources", []))
        if answer_marker.lower() not in answer.lower():
            failures.append((question, f"missing answer marker {answer_marker!r}", answer[:160]))
        if source_marker and source_marker.lower() not in source_text.lower():
            failures.append((question, f"missing source marker {source_marker!r}", source_text))
        print("PASS" if not failures or failures[-1][0] != question else "FAIL", question)

    follow_up = post(
        "Isn't it provisional funds?",
        history=[
            {"role": "user", "content": "What is PF in my payslip?"},
            {"role": "assistant", "content": "PF means Personal Funds."},
        ],
    )
    if "provident fund" not in follow_up.get("answer", "").lower():
        failures.append(("follow-up", "conversation context was not resolved", follow_up.get("answer", "")[:160]))

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(failure)
        return 1
    print(f"\nAll {len(CASES) + 1} live cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
