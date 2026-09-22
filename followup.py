"""Step 2: ask the personal brain what I promised people. Run after ingest.py.

    uv run python followup.py

Implements the CogneeMemory <-> Strands MemoryManager bridge from
https://github.com/sandhya-subramani/Agent-with-a-Brain (agent.py), pointed at
the notes graph built by ingest.py instead of the demo's company docs, and
using Anthropic Claude Sonnet directly (cognee's LLM_PROVIDER=custom +
LiteLLM workaround, same as ingest.py / brain_test.py) instead of Bedrock.

Read top to bottom, same layout as the reference agent.py:
  MEMORY    CogneeMemory: the personal brain, plugged in as a Strands memory store
  SCHEMA    Commitment / Commitments: structured-output extraction target
  TOOLS     draft_followup and send_followup, mock outreach (writes outbox/*.md)
  HOOK      AuditHook prints every tool call
  STEERING  SendApproval asks for a y/n approval before send_followup runs

1. Structured-output extraction of every commitment found in memory.
2. A plain question: "What do I owe people this week?"
3. Draft + send a follow-up to Priya, gated by SendApproval, then record
   completion in memory with add_memory.
4. Ask again: "What do I still owe people?" to confirm it dropped off.
"""

import os
import re
from datetime import date
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
from strands import Agent, tool  # noqa: E402
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry  # noqa: E402
from strands.memory import MemoryManager  # noqa: E402
from strands.memory.types import MemoryEntry  # noqa: E402
from strands.models.anthropic import AnthropicModel  # noqa: E402
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler  # noqa: E402

OUTBOX = ROOT / "outbox"


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


# ── TOOLS ──────────────────────────────────────────────────────────────────────────────
# Mock outreach. No real email is sent; send_followup writes to outbox/<person>.md.

def _outbox_path(person: str) -> Path:
    safe = re.sub(r"[^\w .-]", "", person).strip() or "unknown"
    return OUTBOX / f"{safe}.md"


@tool
def draft_followup(person: str, commitment: str, message: str) -> str:
    """Draft a short follow-up message about a commitment. Does not send anything.

    Args:
        person: Who the message is to.
        commitment: The commitment/promise this message follows up on.
        message: The drafted message text, in English, at most 75 words.
    """
    word_count = len(message.split())
    if word_count > 75:
        return f"Draft rejected: {word_count} words, must be <=75. Shorten it and call draft_followup again."
    return f"Draft ready for {person} ({word_count} words):\n{message}"


@tool
def send_followup(person: str, message: str) -> str:
    """Send a follow-up message to someone. No real email is sent: this writes the
    message to outbox/<person>.md and prints a [SENT] line.

    Args:
        person: Who the message is to.
        message: The message text to send.
    """
    OUTBOX.mkdir(exist_ok=True)
    path = _outbox_path(person)
    with path.open("a") as f:
        f.write(f"## {date.today().isoformat()}\n\n{message}\n\n")
    print(f"\n[SENT] to {person}: {message}")
    return f"Sent to {person}. Saved to {path.relative_to(ROOT)}"


# ── HOOK ───────────────────────────────────────────────────────────────────────────────
# Deterministic code in the agent loop. Runs after every tool call, always.

class AuditHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def after_tool(self, event: AfterToolCallEvent) -> None:
        status = event.result.get("status", "?") if event.result else "cancelled"
        args = event.tool_use.get("input", {})
        print(f"   [AUDIT] {event.tool_use['name']}({args}) -> {status}")


# ── STEERING ───────────────────────────────────────────────────────────────────────────
# Policy that runs BEFORE a tool executes and can redirect the model. No LLM involved.

class SendApproval(SteeringHandler):
    def __init__(self):
        super().__init__(context_providers=[])

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        if tool_use["name"] != "send_followup":
            return Proceed(reason="ok")

        person = tool_use["input"].get("person", "?")
        message = tool_use["input"].get("message", "")
        print(f"\n   [steering] send_followup -> {person}:\n   \"{message}\"")
        answer = input("   Approve? (y/n) ").strip().lower()
        if answer.startswith("y"):
            return Proceed(reason="approved by user")

        feedback = input("   Why not? (reason for the model, optional) ").strip()
        if not feedback:
            feedback = "The user rejected this draft without giving a specific reason."
        print(f"   [steering] BLOCKED send_followup: {feedback}")
        return Guide(
            reason=f"The user rejected this message: {feedback}. Revise it with draft_followup "
            "(still <=75 words), then call send_followup again."
        )


# ── THE AGENT ──────────────────────────────────────────────────────────────────────────

model = AnthropicModel(
    client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
    max_tokens=2048,
    model_id="claude-sonnet-5",
)

agent = Agent(
    model=model,
    tools=[draft_followup, send_followup],
    hooks=[AuditHook()],
    plugins=[SendApproval()],
    memory_manager=MemoryManager(stores=[CogneeMemory()], add_tool_config=True),
    system_prompt=(
        "You are my personal assistant, reviewing my own phone notes about conversations "
        "with people. Facts recalled from memory are things I actually said or promised; "
        "treat them as ground truth and be specific about who and what. "
        "To follow up on a commitment: first call draft_followup with a short (<=75 words) "
        "message, then call send_followup. If send_followup is blocked, read the reason, "
        "revise the message with draft_followup, and call send_followup again. Once a "
        "message is sent successfully, call add_memory with one entry recording that the "
        "commitment is complete, including today's date."
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

    print("\n" + "=" * 78)
    print("Following up on Priya's commitment")
    print("=" * 78)
    today = date.today().isoformat()
    agent(
        "Follow up on the commitment where I promised Priya my LiteLLM workaround writeup. "
        "Draft a short message with draft_followup, then send it with send_followup. If "
        f"it's blocked, revise and try again. Today's date is {today}; once the message is "
        "sent successfully, call add_memory with one entry saying Priya's commitment is "
        f"complete, followed up on {today}."
    )

    print("\n" + "=" * 78)
    print("What do I still owe people?")
    print("=" * 78)
    still_owe = agent("What do I still owe people?")
    print(f"\n{still_owe}")


if __name__ == "__main__":
    main()
