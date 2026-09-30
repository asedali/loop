"""
Ad-hoc connectivity check for whichever LLM provider is configured.

Kept out of `app/` on purpose: this is a scratch script for poking the
provider by hand, not part of the request path. Read credentials from the
environment (or the repo .env) — never hardcode a key here.
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://opencode.ai/zen/v1"
).rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "glm-5.3")

if not API_KEY:
    raise SystemExit(
        "No API key found. Set LLM_API_KEY (or OPENROUTER_API_KEY) in the "
        "environment or in .env before running this script."
    )

MESSAGES = [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]


def ask(messages, headers=None):
    response = requests.post(
        url=f"{BASE_URL}/chat/completions",
        headers=headers
        or {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"model": MODEL, "messages": messages}),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


first = ask(MESSAGES)
print(f"model: {MODEL}")
print(f"reply: {first.get('content')}")

# Second turn, replaying the reasoning trace if the provider returned one.
followup = ask(
    [
        *MESSAGES,
        {
            "role": "assistant",
            "content": first.get("content"),
            "reasoning_details": first.get("reasoning_details"),
        },
        {"role": "user", "content": "Are you sure? Think carefully."},
    ]
)
print(f"reply: {followup.get('content')}")
