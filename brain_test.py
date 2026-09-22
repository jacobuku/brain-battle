"""Prove that cognee memory persists across separate Strands agent instances.

Uses a Strands agent (Anthropic Claude Sonnet model) with cognee_tools() attached:
  1. Agent #1 remembers a fact.
  2. A brand-new Agent #2 instance (fresh process-level object, no shared state)
     recalls that fact, showing the memory lives in cognee's own store rather
     than in the agent's in-memory conversation.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# cognee reads its LLM key from LLM_API_KEY. We reuse ANTHROPIC_API_KEY from
# .env at runtime instead of duplicating the secret into .env or any file.
os.environ.setdefault("LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])

from strands import Agent
from strands.models.anthropic import AnthropicModel
from cognee_integration_strands import cognee_tools


def make_agent() -> Agent:
    model = AnthropicModel(
        client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
        max_tokens=1024,
        model_id="claude-sonnet-5",
    )
    return Agent(model=model, tools=cognee_tools())


def main() -> None:
    fact = "The secret project codename is Brain Battle and its budget is $42,000."

    print("=== Agent #1: remembering ===")
    agent_one = make_agent()
    remember_response = agent_one(f"Remember that: {fact}")
    print(remember_response)

    print("\n=== Agent #2 (new instance): recalling ===")
    agent_two = make_agent()
    recall_response = agent_two("What is the secret project codename and its budget?")
    print(recall_response)


if __name__ == "__main__":
    main()
