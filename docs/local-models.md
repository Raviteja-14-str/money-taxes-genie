# Local model learning paths

Money & Taxes Genie keeps the same browser contract and safety prompt across
Bedrock, Ollama, and Hugging Face. Run `local_api.py` from this project folder.

## Path 1: Ollama — easiest fully local path

Ollama runs the model on your Mac and exposes a local chat endpoint. It does not
need a model API key, and the model files stay on the machine.

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Download and test a small multilingual model:

   ```zsh
   ollama run qwen3:4b
   ```

3. In another terminal, start Money Genie:

   ```zsh
   MG_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b python3 local_api.py
   ```

4. Open `http://127.0.0.1:3000`.

For a smaller laptop footprint, try `qwen3:1.7b` or `gemma3:1b`. The model
names and sizes are listed in the Ollama libraries for [Qwen3](https://ollama.com/library/qwen3)
and [Gemma 3](https://ollama.com/library/gemma3).

## Path 2: Hugging Face Transformers — learn the model stack

This path downloads model weights from the Hugging Face Hub and runs them in
the Python process. It is useful for learning tokenizers, model loading, and
generation directly. The first run can be slow and needs disk/RAM.

```zsh
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install -r local/requirements-hf.txt
MG_PROVIDER=hf_local HF_MODEL=Qwen/Qwen3-0.6B python local_api.py
```

The `Qwen/Qwen3-0.6B` model is intentionally small for the first lesson. Move
to `Qwen/Qwen3-1.7B` or a larger model after the pipeline works. Keep the model
download outside the repository; never commit model weights.

## Optional Path 3: Hugging Face hosted inference

This keeps your server code local but sends prompts to Hugging Face. It needs a
Hugging Face token in the shell environment and may have provider limits or
charges, so it is not the zero-cost local path:

```zsh
export HF_TOKEN='paste-token-only-in-your-terminal'
MG_PROVIDER=hf_api HF_MODEL=Qwen/Qwen3-0.6B python local_api.py
```

Do not put `HF_TOKEN` in `.env`, frontend code, GitHub, or browser storage.

## What you learn from the three paths

| Path | Where the model runs | Key lesson | Cost shape |
| --- | --- | --- | --- |
| Bedrock | AWS | Managed production inference and IAM | Metered AWS usage/credits |
| Ollama | Your Mac | Local serving and HTTP APIs | No model API bill; uses local compute |
| Hugging Face local | Your Mac/Python | Tokenizer, weights, generation pipeline | No model API bill; uses disk/RAM |
| Hugging Face hosted | HF provider | Hosted inference authentication | Token/provider limits or charges |

All paths use the same small local retrieval layer here. That makes model
comparisons meaningful: change the provider, not the knowledge or safety rules.
