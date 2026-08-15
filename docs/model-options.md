# Model options for the Money & Taxes Genie POC

## Recommended order

1. **Amazon Nova Micro** — `amazon.nova-micro-v1:0`
   - Simplest Bedrock alternative to Anthropic for this POC.
   - Uses Bedrock's model-agnostic Converse API.
   - Does not require Anthropic's first-time-use form.
   - Choose this first after the account-level Bedrock restriction is cleared.

2. **Claude 3 Haiku** — `anthropic.claude-3-haiku-20240307-v1:0`
   - Current default because it is compact and suitable for short educational answers.
   - Requires the Anthropic use-case form before first use.

3. **Amazon Nova Lite** — `amazon.nova-lite-v1:0`
   - A stronger Amazon option for comparing answer quality after the POC works.
   - Uses the same Converse path as Nova Micro.

## Cost note

Bedrock inference is metered; there is no promise that a model is permanently free.
Use the account's AWS credits or Free Tier eligibility, set a budget alert, and
keep the POC's prompts and output limits small. Check current prices before
opening the public endpoint: https://aws.amazon.com/bedrock/pricing/

For a genuinely no-inference-cost experiment, run a local model such as Qwen or
Gemma through a local runtime. That is useful for development, but it cannot be
used directly by the deployed Lambda without changing the architecture.

## Switching models

Set `CHAT_MODEL_ID` on the Lambda. The code selects the correct request shape for
Claude or Nova. The separate Titan embedding model is still required for the
current retrieval design, so changing only the answer model does not bypass an
embedding-model access or quota problem.
