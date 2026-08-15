"""
Money & Taxes Genie — Ingestion Script
----------------------------------------
What this does, in plain terms:
  1. Reads every .txt file from your S3 bucket
  2. Cuts each one into paragraph-sized chunks
  3. Sends each chunk to Bedrock's Titan Embeddings model, which turns the
     text into a list of numbers (a "fingerprint" capturing its meaning)
  4. Saves {chunk text, fingerprint, source doc} into your DynamoDB table

Run this once now, and again any time you add or change documents in S3.
"""

import boto3
import json
import uuid
from decimal import Decimal

# ---------- EDIT THESE TWO LINES ----------
BUCKET_NAME = "fingenie-docs-2026"   # <-- put your actual S3 bucket name from Step 1
TABLE_NAME = "MoneyGenieChunks"          # <-- matches what you created in Step 2b
# -------------------------------------------

REGION = "ap-south-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


def chunk_text(text, max_chars=700):
    """Split into paragraphs, then further split any long paragraph by sentence."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            sentences = p.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|")
            cur = ""
            for s in sentences:
                if len(cur) + len(s) > max_chars and cur:
                    chunks.append(cur.strip())
                    cur = s
                else:
                    cur += (" " if cur else "") + s
            if cur:
                chunks.append(cur.strip())
    return chunks


def embed(text):
    """Call Bedrock Titan Embeddings and return the fingerprint (list of floats)."""
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def to_decimal_list(vec):
    """DynamoDB needs Decimal, not native float, for numbers."""
    return [Decimal(str(round(v, 8))) for v in vec]


def get_doc_title(text, fallback):
    """Use the first 'TITLE: ...' line if present, else the filename."""
    first_line = text.strip().split("\n")[0]
    if first_line.upper().startswith("TITLE:"):
        return first_line.split(":", 1)[1].strip()
    return fallback


def main():
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME)
    if "Contents" not in objects:
        print("No files found in bucket. Did you upload the .txt files?")
        return

    total_chunks = 0
    for obj in objects["Contents"]:
        key = obj["Key"]
        if not key.endswith(".txt"):
            continue

        print(f"Processing {key} ...")
        body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
        doc_title = get_doc_title(body, fallback=key)
        chunks = chunk_text(body)

        for chunk in chunks:
            fingerprint = embed(chunk)
            table.put_item(Item={
                "chunk_id": str(uuid.uuid4()),
                "doc_title": doc_title,
                "doc_source": key,
                "chunk_text": chunk,
                "embedding": to_decimal_list(fingerprint),
            })
            total_chunks += 1

        print(f"  -> {len(chunks)} chunks embedded and stored.")

    print(f"\nDone. {total_chunks} total chunks now in {TABLE_NAME}.")


if __name__ == "__main__":
    main()
