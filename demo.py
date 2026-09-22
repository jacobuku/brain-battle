"""Judge-facing demo: chains prebrief.py + ingest.py + followup.py into one run.

    uv run python demo.py

Feature freeze: no new capabilities here, just the existing logic from the
three scripts, in one scripted narrative with three stages:

  [1/3] BEFORE THE EVENT  -- prebrief.py's fetch_page + memory briefing
  [2/3] AFTER THE EVENT   -- followup.py's structured-output commitment
                             extraction, printed as a table
  [3/3] TAKE ACTION       -- followup.py's draft -> approve(y/n) -> send ->
                             add_memory -> re-ask flow, for the first
                             commitment from stage 2

Keeps the same security boundary added to prebrief.py: the agent that calls
fetch_page (untrusted, scraped web content) never has add_memory; the agent
that can write memory never has fetch_page.
"""

import os
import re
import time
from pathlib import Path

import requests
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

EVENT_URL = "https://luma.com/sep-21-cognee-aws"
BRIGHTDATA_ZONE = "web_unlocker1"
OUTBOX = ROOT / "outbox"
BRAIN_HTML = ROOT / "brain.html"
REAL_NOTES = ROOT / "data" / "real_notes.txt"
SAMPLE_NOTES = ROOT / "data" / "sample_notes.txt"


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ── MEMORY (from followup.py / prebrief.py) ──────────────────────────────────────────

class CogneeMemory:
    name = "personal_brain"
    description = "My phone notes and project facts: people, promises, and things I've learned."
    max_search_results = 10
    writable = True
    extraction = None  # required by this strands-agents version's MemoryStore protocol

    async def search(self, query, options=None):
        print(f"\n   [memory] cognee.recall({query[:80]!r}{'...' if len(query) > 80 else ''})")
        results = await cognee.recall(query, top_k=self.max_search_results)
        return [MemoryEntry(content=r.text) for r in results if getattr(r, "text", None)]

    async def add(self, content, metadata=None):
        preview = content[:80] + ("..." if len(content) > 80 else "")
        print(f"\n   [memory] cognee.remember({preview!r})")
        await cognee.remember(content, self_improvement=False)


# ── STAGE 1 TOOL: fetch_page (from prebrief.py, with its security fixes) ────────────

ALLOWED_FETCH_DOMAINS = {"luma.com", "brightdata.com", "cognee.ai", "aws.amazon.com", "docker.com"}


def _is_allowed_host(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_FETCH_DOMAINS)


@tool
def fetch_page(url: str) -> str:
    """Fetch a single public webpage and return its visible text. Restricted to a fixed
    domain allowlist (the event site and known host/sponsor companies).

    Args:
        url: The page to fetch.
    """
    if not _is_allowed_host(url):
        return f"Refused: {url!r} is not on the allowed domain list. Not fetched."

    token = os.environ["BRIGHTDATA_API_TOKEN"]
    resp = requests.post(
        "https://api.brightdata.com/request",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"zone": BRIGHTDATA_ZONE, "url": url, "format": "raw"},
        timeout=60,
    )
    resp.raise_for_status()
    html = resp.text
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    import html as htmlmod

    text = htmlmod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:20000]


# ── STAGE 3 TOOLS: draft_followup / send_followup (from followup.py) ─────────────────

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
    """Send a follow-up message to someone. No real email is sent: writes to
    outbox/<person>.md and prints a [SENT] line.

    Args:
        person: Who the message is to.
        message: The message text to send.
    """
    from datetime import date

    OUTBOX.mkdir(exist_ok=True)
    path = _outbox_path(person)
    with path.open("a") as f:
        f.write(f"## {date.today().isoformat()}\n\n{message}\n\n")
    print(f"\n[SENT] to {person}: {message}")
    return f"Sent to {person}. Saved to {path.relative_to(ROOT)}"


class AuditHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def after_tool(self, event: AfterToolCallEvent) -> None:
        status = event.result.get("status", "?") if event.result else "cancelled"
        args = event.tool_use.get("input", {})
        print(f"   [AUDIT] {event.tool_use['name']}({args}) -> {status}")


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


# ── SCHEMAS (from prebrief.py / followup.py) ──────────────────────────────────────────

class Organization(BaseModel):
    name: str = Field(description="Company name")
    role: str = Field(description='"host" or "sponsor"')
    website: str | None = Field(default=None, description="Company website/domain, if known")


class EventOrganizations(BaseModel):
    items: list[Organization] = Field(description="Every host/sponsor company on the page")


class Commitment(BaseModel):
    person: str = Field(description="Who I made the promise to")
    company: str | None = Field(default=None, description="Their company/organization, if mentioned")
    what_i_promised: str = Field(description="What I said I would do")
    due: str | None = Field(default=None, description="When it's due, if a date or timeframe was mentioned")
    context: str = Field(description="Brief context of the conversation this came from")


class Commitments(BaseModel):
    items: list[Commitment] = Field(description="All commitments/promises found in my notes")


# ── AGENTS ─────────────────────────────────────────────────────────────────────────────
# Two agents, same split as prebrief.py/followup.py: the one that fetches untrusted web
# content never gets add_memory; the one that can write memory never gets fetch_page.

model = AnthropicModel(
    client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
    max_tokens=2048,
    model_id="claude-sonnet-5",
)

brief_agent = Agent(
    model=model,
    tools=[fetch_page],
    memory_manager=MemoryManager(stores=[CogneeMemory()], add_tool_config=False),
    system_prompt=(
        "You help me prep for events. Facts recalled from memory are things I actually "
        "know or promised; treat them as ground truth. When asked about an event page, "
        "use fetch_page on the event URL only -- never fetch an individual's profile page, "
        "only event pages and company sites."
    ),
)

action_agent = Agent(
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


def print_commitments_table(items: list[Commitment]) -> None:
    rows = [(c.person, c.company or "-", c.what_i_promised, c.due or "-") for c in items]
    headers = ("Person", "Company", "Promised", "Due")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(4)
    ]
    widths[2] = min(widths[2], 60)  # keep "Promised" column readable

    def fmt_row(row):
        cells = []
        for i, val in enumerate(row):
            val = (val[: widths[2] - 1] + "…") if i == 2 and len(val) > widths[2] else val
            cells.append(val.ljust(widths[i]))
        return " | ".join(cells)

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    start = time.time()

    notes_source = "REAL NOTES" if REAL_NOTES.exists() else "SAMPLE DATA"
    print(f"Notes source: {notes_source}")

    # ── [1/3] BEFORE THE EVENT ───────────────────────────────────────────────────────
    banner("[1/3] BEFORE THE EVENT")
    result = brief_agent(
        f"Fetch {EVENT_URL} with fetch_page. From the page text, list every host and "
        "sponsor/technology-partner COMPANY (not individual people) mentioned. Skip "
        "individual names.",
        structured_output_model=EventOrganizations,
    )
    orgs = result.structured_output
    print(f"\nFound {len(orgs.items)} companies:\n")
    for o in orgs.items:
        site = f" ({o.website})" if o.website else ""
        print(f"- {o.name} [{o.role}]{site}")

    company_list = ", ".join(f"{o.name} ({o.role})" for o in orgs.items)
    briefing = brief_agent(
        f"Tonight's event companies: {company_list}. Using everything in memory, write a "
        "short briefing: for each company, who I should try to find tonight and 2-3 "
        "specific things to talk to them about."
    )
    print(f"\n{briefing}")

    # ── [2/3] AFTER THE EVENT ────────────────────────────────────────────────────────
    banner("[2/3] AFTER THE EVENT")
    result = action_agent(
        "Search my notes for every promise or commitment I made to someone. List all of "
        "them, don't skip any.",
        structured_output_model=Commitments,
    )
    commitments = result.structured_output
    print(f"\nFound {len(commitments.items)} commitment(s):\n")
    print_commitments_table(commitments.items)

    # ── [3/3] TAKE ACTION ────────────────────────────────────────────────────────────
    banner("[3/3] TAKE ACTION")
    if not commitments.items:
        print("\nNo commitments found -- nothing to act on.")
    else:
        first = commitments.items[0]
        print(f"\nActing on the first commitment: {first.person} -- {first.what_i_promised}")
        action_agent(
            f"Follow up on the commitment where I promised {first.person}: "
            f"{first.what_i_promised}. Draft a short message with draft_followup, then "
            "send it with send_followup. If it's blocked, revise and try again. Once the "
            "message is sent successfully, call add_memory with one entry saying "
            f"{first.person}'s commitment is complete, with today's date."
        )

        print("\n" + "-" * 78)
        print("What do I still owe people?")
        print("-" * 78)
        still_owe = action_agent("What do I still owe people?")
        print(f"\n{still_owe}")

    # ── WRAP-UP ──────────────────────────────────────────────────────────────────────
    # Not regenerating brain.html here: cognee.visualize_graph() needs a graph-engine
    # connection scoped the same way recall/remember are, and calling it from a fresh,
    # unrelated asyncio.run() after the agent calls above returned "No nodes found in
    # the database" -- it silently wrote an EMPTY graph over the good one ingest.py
    # built. Rather than debug cognee's per-loop/per-session graph-engine scoping under
    # a feature freeze, just point at the file ingest.py already produced.
    print(f"\nKnowledge graph: {BRAIN_HTML}")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
