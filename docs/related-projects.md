# Related projects reviewed

Updated: 2026-08-15

These projects were reviewed for architecture and safety patterns. Their content and code were not copied into Money Genie.

- [itr-wala](https://github.com/karanb192/itr-wala): separates language-model document reading from deterministic, tested tax computation. Useful lesson: an LLM should explain and gather facts; code should calculate rates, rebates, and interest.
- [india-itr-copilot](https://github.com/Loki200399/india-itr-copilot): uses a documents-first workflow, reconciliation, and source-attributed fields. Useful lesson: extract from Form 16/AIS/26AS first and ask only for missing facts.
- [FinanceParam](https://huggingface.co/bharatgenai/FinanceParam): demonstrates India-focused, bilingual financial knowledge. Useful lesson: Hindi and other Indian-language coverage should be added only with source review and evaluation, not by assuming translation preserves legal meaning.
- [Indian personal-finance dataset](https://github.com/Arif-miad/Indian-Personal-Finance-and-Spending-Habits): illustrates budgeting and spending categories. It is a research dataset, not authoritative financial guidance, and should not be treated as real user data.

The reusable design pattern is therefore:

`authoritative sources -> dated knowledge files -> retrieval -> local model explanation`

For tax arithmetic, investment calculations, and eligibility checks, use deterministic code plus current official sources rather than asking the language model to invent numbers.
