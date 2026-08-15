"""
Money & Taxes Genie — Answer Lambda
-------------------------------------
What this does, in plain terms:
  1. Receives a question from your chat frontend
  2. Turns the question into a "fingerprint" (embedding), same way we did for the docs
  3. Compares that fingerprint against every chunk fingerprint in DynamoDB,
     using cosine similarity (a standard way to measure "how close two
     fingerprints point in the same direction" — 1.0 = identical meaning,
     0 = unrelated)
  4. Takes the top 5 matching chunks and hands them to Claude as context
  5. Returns Claude's answer + which sources it used
"""

import json
import math
import os
import boto3

REGION = os.environ.get("AWS_REGION", "ap-south-1")
TABLE_NAME = os.environ.get("TABLE_NAME", "MoneyGenieChunks")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Keep the model configurable so a model change never requires editing code.
# Haiku 3 is an inexpensive on-demand starting point for this learning MVP.
CHAT_MODEL_ID = os.environ.get(
    "CHAT_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",
)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

RELEVANCE_THRESHOLD = 0.3  # chunks below this similarity score are ignored

dynamodb = boto3.resource("dynamodb", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

SYSTEM_PROMPT = """You are Money & Taxes Genie, a plain-language assistant that explains Indian personal finance and tax concepts.
Rules:
- Answer ONLY using the CONTEXT provided below.
- If the context doesn't cover the question, say so plainly instead of guessing.
- This is general education, never personalized financial or tax advice — don't tell the user what they specifically should do with their money.
- Keep answers clear and to the point."""


def get_embedding(text):
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_chunks(question_embedding, top_k=5):
    # A full table scan is fine at this scale (a few dozen chunks).
    # If your knowledge base grows into the thousands, this is the part
    # you'd eventually swap for a real vector database.
    items = table.scan().get("Items", [])
    scored = []
    for item in items:
        chunk_embedding = [float(x) for x in item["embedding"]]
        score = cosine_similarity(question_embedding, chunk_embedding)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [pair for pair in scored[:top_k] if pair[0] >= RELEVANCE_THRESHOLD]


def ask_model(question, context_text):
    user_message = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"
    # Amazon Nova uses Bedrock's model-agnostic Converse API. Keeping this
    # branch here lets the POC switch between Claude Haiku and Nova Micro/Lite
    # with only the CHAT_MODEL_ID environment variable.
    if CHAT_MODEL_ID.startswith("amazon.nova"):
        response = bedrock.converse(
            modelId=CHAT_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 1000},
        )
        return response["output"]["message"]["content"][0]["text"]

    response = bedrock.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}]
        })
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def make_response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": json.dumps(body_dict),
    }


def lambda_handler(event, context):
    try:
        raw_body = event.get("body") or "{}"
        body = json.loads(raw_body)
        question = (body.get("question") or "").strip()

        if not question:
            return make_response(400, {"error": "No question provided."})
        if len(question) > 4000:
            return make_response(413, {"error": "Question is too long."})

        question_embedding = get_embedding(question)
        top_chunks = retrieve_top_chunks(question_embedding)

        if top_chunks:
            context_text = "\n\n---\n\n".join(
                f"[Source: {item['doc_title']}]\n{item['chunk_text']}"
                for score, item in top_chunks
            )
            answer = ask_model(question, context_text)
            sources = [
                {"doc_title": item["doc_title"], "relevance": round(score, 3)}
                for score, item in top_chunks
            ]
        else:
            answer = ask_model(question, "(no relevant sources found in the knowledge base)")
            sources = []

        return make_response(200, {"answer": answer, "sources": sources})

    except Exception:
        # Do not expose AWS, model, table, or implementation details through a
        # public Function URL.
        return make_response(500, {"error": "Money & Taxes Genie is temporarily unavailable."})
