"""
llm_client.py

Purpose & Use:
Centralized LLM client factory and model configuration for the Text-to-SQL system.
Manages the OpenRouter API connection using the OpenAI Python SDK.

Why this file exists:
- OpenRouter exposes an OpenAI-compatible API, so this module configures the standard
  `openai.OpenAI` SDK with OpenRouter's base URL (`https://openrouter.ai/api/v1`).
- Provides a singleton client via `get_llm_client()` initialized with `OPENROUTER_API_KEY`.
- Defines `MODEL_NAME` in a single location so the LLM model can be updated across the entire
  application without modifying individual agent nodes.

Where it is used:
- `generate_sql.py`: Calls the LLM to translate natural language questions into SQL.
- `synthesize_response.py`: Calls the LLM to synthesize conversational answers from query result rows.

Testing:
Run directly to verify your OpenRouter API key and connectivity:
    python llm_client.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Free, tool-use/reasoning-capable model on OpenRouter.
#
# NOTE: OpenRouter's individual :free model slugs churn constantly -- two
# different ones (z-ai/glm-5.2:free, then openai/gpt-oss-120b:free) got
# pulled to paid-only within the same week of testing this. Rather than
# hardcoding one specific model and having it break again, this uses
# OpenRouter's own free-model router, which automatically selects from
# whatever free models are currently available and still supports tool
# calling / structured outputs. If you ever want to pin a specific model
# again instead, just replace this string.
MODEL_NAME = "openrouter/free"

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return _client


if __name__ == "__main__":
    # Quick sanity check: run `python llm_client.py` to confirm the key works.
    client = get_llm_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
    )
    print("Model replied:", response.choices[0].message.content)