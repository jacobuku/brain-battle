"""Step 2: ask the personal brain what I promised people. Run after ingest.py.

    uv run python followup.py

Implements the CogneeMemory <-> Strands MemoryManager bridge from
https://github.com/sandhya-subramani/Agent-with-a-Brain (agent.py), pointed at
the notes graph built by ingest.py instead of the demo's company docs, and
using Anthropic Claude Sonnet directly (cognee's LLM_PROVIDER=custom +
LiteLLM workaround, same as ingest.py / brain_test.py) instead of Bedrock.

1. Structured-output extraction of every commitment found in memory.
2. A plain question: "What do I owe people this week?"
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

os.environ.setdefault("LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(ROOT / ".data_storage"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(ROOT / ".cognee_system"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(ROOT / ".cognee_cache"))

import cognee  # noqa: E402
from strands import Agent  # noqa: E402
from strands.memory import MemoryManager  # noqa: E402
from strands.memory.types import MemoryEntry  # noqa: E402
from strands.models.anthropic import AnthropicModel  # noqa: E402


# ── MEMORY ─────────────────────────────────────────────────────────────────────────────
# Strands' MemoryManager works with any object that has `search` and `add`.
# Cognee's recall and remember are exactly those two operations.

class CogneeMemory:
    name = "personal_brain"
    description = "My phone notes: conversations with people and what I promised them."
    max_search_results = 10
    writable = True
    extraction = None  # required by this strands-agents version's MemoryStore protocol;
    # we run our own explicit structured-output extraction below instead of automatic.

    async def search(self, query, options=None):
        print(f"\n   [memory] cognee.recall({query!r})")
        results = await cognee.recall(query, top_k=self.max_search_results)
        return [MemoryEntry(content=r.text) for r in results if getattr(r, "text", None)]

    async def add(self, content, metadata=None):
        print(f"\n   [memory] cognee.remember({content!r})")
        await cognee.remember(content, self_improvement=False)


# ── STRUCTURED OUTPUT SCHEMA ──────────────────────────────────────────────────────────

class Commitment(BaseModel):
    """A single promise I made to someone, found in my notes."""

    person: str = Field(description="Who I made the promise to")
    company: str | None = Field(default=None, description="Their company/organization, if mentioned")
    what_i_promised: str = Field(description="What I said I would do")
    due: str | None = Field(default=None, description="When it's due, if a date or timeframe was mentioned")
    context: str = Field(description="Brief context of the conversation this came from")


class Commitments(BaseModel):
    """Every commitment found in memory."""

    items: list[Commitment] = Field(description="All commitments/promises found in my notes")


# ── THE AGENT ──────────────────────────────────────────────────────────────────────────

model = AnthropicModel(
    client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
    max_tokens=2048,
    model_id="claude-sonnet-5",
)

agent = Agent(
    model=model,
    memory_manager=MemoryManager(stores=[CogneeMemory()], add_tool_config=True),
    system_prompt=(
        "You are my personal assistant, reviewing my own phone notes about conversations "
        "with people. Facts recalled from memory are things I actually said or promised; "
        "treat them as ground truth and be specific about who and what."
    ),
)


def main() -> None:
    print("=" * 78)
    print("Extracting every commitment from memory (structured output)")
    print("=" * 78)
    result = agent(
        "Search my notes for every promise or commitment I made to someone. List all of "
        "them, don't skip any.",
        structured_output_model=Commitments,
    )
    commitments = result.structured_output
    print(f"\nFound {len(commitments.items)} commitment(s):\n")
    for i, c in enumerate(commitments.items, start=1):
        who = f"{c.person} ({c.company})" if c.company else c.person
        due = f" -- due {c.due}" if c.due else ""
        print(f"{i}. {who}: {c.what_i_promised}{due}")
        print(f"   context: {c.context}")

    print("\n" + "=" * 78)
    print("What do I owe people this week?")
    print("=" * 78)
    answer = agent("What do I owe people this week?")
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
