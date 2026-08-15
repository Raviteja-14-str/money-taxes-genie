# Money & Taxes Genie

Local-first Indian personal finance and tax literacy assistant using retrieval, source links, and safety boundaries.

## Run locally

The browser talks to the local API; provider credentials never enter browser code.

```sh
MG_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b MAX_OUTPUT_TOKENS=300 python3 local_api.py
```

The Hugging Face local provider is also available after its environment is installed:

\`\`\`sh
LOCAL_PORT=3001 MG_PROVIDER=hf_local HF_MODEL=Qwen/Qwen3-0.6B MAX_OUTPUT_TOKENS=220 \
  .venv-hf/bin/python local_api.py
\`\`\`

Open `http://localhost:3000`.

## Verify changes

Fast checks:

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile local_api.py lambda_function.py tests/live_eval.py
node --check web/app.js
git diff --check
```

With the Ollama server running, the synthetic live suite exercises common user questions, follow-ups, unsupported topics, source attribution, and protected actions:

```sh
python3 tests/live_eval.py
```

The assistant is educational only. It does not access accounts, perform transactions, file returns, or request PAN, Aadhaar, passwords, card details, or OTPs.
