"""Provider-neutral tool-calling support for Gemini and Groq.

API keys are read from Streamlit secrets or environment variables and are never
returned by this module. Both providers expose an OpenAI-compatible chat API,
which lets the same bounded scientific tool loop work for either provider.
"""

import json
import os
from typing import Any, Callable


PROVIDERS = {
    "Gemini": {
        "secret_name": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash"],
    },
    "Groq": {
        "secret_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-20b",
        "models": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
        ],
    },
}


def get_configured_key(provider: str) -> str | None:
    """Read one provider key from Streamlit secrets or the environment."""
    config = PROVIDERS.get(provider)
    if not config:
        return None
    key = os.getenv(config["secret_name"])
    try:
        import streamlit as st
        secret_value = st.secrets.get(config["secret_name"])
        if secret_value:
            key = str(secret_value)
    except Exception:
        pass
    return key or None


def provider_status() -> dict[str, bool]:
    """Return configuration status without exposing key values."""
    return {provider: bool(get_configured_key(provider)) for provider in PROVIDERS}


def make_client(provider: str, api_key: str | None = None):
    """Create an OpenAI-compatible client for Gemini or Groq."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    from openai import OpenAI
    key = api_key or get_configured_key(provider)
    if not key:
        raise RuntimeError(f"{provider} API key is not configured.")
    return OpenAI(api_key=key, base_url=PROVIDERS[provider]["base_url"])


def run_tool_calling_agent(
    provider: str,
    model: str,
    question: str,
    tool_definitions: list[dict],
    dispatch: Callable[[str, dict], dict],
    api_key: str | None = None,
    max_rounds: int = 6,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a bounded multi-round tool-calling loop."""
    client = make_client(provider, api_key=api_key)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a careful climate scientist and offshore wind analyst. "
                "Use the supplied tools, explain your plan through tool calls, and cite the returned values. "
                "Never call daily January data bankable, never claim direct 100 m wind when the data is 10 m, "
                "and clearly state when a requested period is unavailable."
            ),
        },
        {"role": "user", "content": question},
    ]
    trace = [{"step": 1, "action": "LLM created a bounded analysis plan", "provider": provider, "model": model}]
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            trace.append({"step": len(trace) + 1, "action": "LLM synthesized tool results"})
            return message.content or "The agent returned no text.", trace
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            trace.append({"step": len(trace) + 1, "action": "Call bounded scientific tool", "tool": call.function.name, "arguments": arguments})
            result = dispatch(call.function.name, arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    return "The agent reached its maximum tool-call rounds without a final answer.", trace
