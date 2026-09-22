"""Pre-event briefing: who to find tonight and what to talk to them about.

    uv run python prebrief.py

Reuses the CogneeMemory <-> Strands MemoryManager bridge from followup.py
(itself from https://github.com/sandhya-subramani/Agent-with-a-Brain), plus
Anthropic Claude Sonnet + cognee's LLM_PROVIDER=custom + LiteLLM + fastembed
setup from ingest.py / followup.py / brain_test.py.

New here: a fetch_page(url) tool that scrapes a single public page's text via
Bright Data's Web Unlocker REST API (https://docs.brightdata.com), using
BRIGHTDATA_API_TOKEN and the "web_unlocker1" zone. No MCP server: the docs'
own fallback ("if MCP setup is troublesome, write a @tool against Web
Unlocker/scrape API directly") since a bare @tool calling one REST endpoint
is far less moving parts than standing up a local MCP server subprocess.

1. fetch_page(https://luma.com/sep-21-cognee-aws) -> host/sponsor companies,
   via structured output. Only the event page itself is fetched -- no
   individual profile pages (e.g. luma.com/user/<name>).
2. Two facts about this project's own cognee/Strands debugging are added to
   memory (so the briefing can draw on them like anything else recalled).
3. A briefing: which company's people to find tonight, and what to talk to
   each about, combining the event page with memory.
"""

import os
import re
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
from strands.memory import MemoryManager  # noqa: E402
from strands.memory.types import MemoryEntry  # noqa: E402
from strands.models.anthropic import AnthropicModel  # noqa: E402

BRIGHTDATA_ZONE = "web_unlocker1"
EVENT_URL = "https://luma.com/sep-21-cognee-aws"


# ── MEMORY ─────────────────────────────────────────────────────────────────────────────
# Same bridge as followup.py: Strands' MemoryManager works with any object that has
# `search` and `add`; cognee's recall and remember are exactly those two operations.

class CogneeMemory:
    name = "personal_brain"
    description = "My phone notes and project facts: people, promises, and things I've learned."
    max_search_results = 10
    writable = True
    extraction = None  # required by this strands-agents version's MemoryStore protocol

    async def search(self, query, options=None):
        print(f"\n   [memory] cognee.recall({query!r})")
        results = await cognee.recall(query, top_k=self.max_search_results)
        return [MemoryEntry(content=r.text) for r in results if getattr(r, "text", None)]

    async def add(self, content, metadata=None):
        print(f"\n   [memory] cognee.remember({content!r})")
        await cognee.remember(content, self_improvement=False)


# ── TOOL ───────────────────────────────────────────────────────────────────────────────
# Bright Data Web Unlocker: POST url+zone, get back the rendered page. One REST call,
# no MCP server process to manage. https://docs.brightdata.com/products/web-unlocker

@tool
def fetch_page(url: str) -> str:
    """Fetch a single public webpage and return its visible text (scripts/styles/tags
    stripped). Only use this on public pages like event listings or company websites --
    never on individual profile pages.

    Args:
        url: The page to fetch.
    """
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    return text[:20000]


# ── SCHEMA ─────────────────────────────────────────────────────────────────────────────

class Organization(BaseModel):
    """A company found on the event page, as a host or sponsor."""

    name: str = Field(description="Company name")
    role: str = Field(description='"host" or "sponsor" (how they relate to the event)')
    website: str | None = Field(default=None, description="Company website/domain, if known")


class EventOrganizations(BaseModel):
    items: list[Organization] = Field(description="Every host/sponsor company on the page")


# ── THE AGENT ──────────────────────────────────────────────────────────────────────────

model = AnthropicModel(
    client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
    max_tokens=2048,
    model_id="claude-sonnet-5",
)

agent = Agent(
    model=model,
    tools=[fetch_page],
    memory_manager=MemoryManager(stores=[CogneeMemory()], add_tool_config=True),
    system_prompt=(
        "You help me prep for events. Facts recalled from memory are things I actually "
        "know or promised; treat them as ground truth. When asked about an event page, "
        "use fetch_page on the event URL only -- never fetch an individual's profile page "
        "(e.g. a /user/... or personal LinkedIn URL), only event pages and company sites."
    ),
)


def main() -> None:
    # Seed two facts from this project's own build so the briefing can draw on them.
    print("=" * 78)
    print("Adding two build-notes to memory")
    print("=" * 78)
    import asyncio

    asyncio.run(
        cognee.remember(
            "Debugging note: cognee 1.1.2's built-in LLM_PROVIDER=anthropic adapter is "
            "broken -- it calls instructor.patch() without provider=Provider.ANTHROPIC, so "
            "instructor defaults to Provider.OPENAI and rejects the anthropic_tools mode. "
            "Workaround: use LLM_PROVIDER=custom, which routes through LiteLLM instead.",
            self_improvement=False,
        )
    )
    asyncio.run(
        cognee.remember(
            "Debugging note: this project's strands-agents version added a required "
            "`extraction` attribute to the MemoryStore protocol (not present in the "
            "Agent-with-a-Brain reference repo's CogneeMemory). Any custom memory store "
            "passed to MemoryManager needs `extraction = None` or it raises AttributeError.",
            self_improvement=False,
        )
    )

    print("\n" + "=" * 78)
    print(f"Fetching event page: {EVENT_URL}")
    print("=" * 78)
    result = agent(
        f"Fetch {EVENT_URL} with fetch_page. From the page text, list every host and "
        "sponsor/technology-partner COMPANY (not individual people) mentioned -- for "
        "example who it's \"Presented by\", \"Hosted By\", and any companies named in an "
        "agentic stack / sponsors / credits section. Skip individual names.",
        structured_output_model=EventOrganizations,
    )
    orgs = result.structured_output
    print(f"\nFound {len(orgs.items)} companies:\n")
    for o in orgs.items:
        site = f" ({o.website})" if o.website else ""
        print(f"- {o.name} [{o.role}]{site}")

    print("\n" + "=" * 78)
    print("Tonight's briefing")
    print("=" * 78)
    company_list = ", ".join(f"{o.name} ({o.role})" for o in orgs.items)
    briefing = agent(
        f"Tonight's event companies: {company_list}. Using everything in memory -- my "
        "notes, promises, and debugging learnings -- write a short briefing: for each "
        "company, who I should try to find tonight and 2-3 specific things to talk to "
        "them about. Prioritize the companies I actually have something concrete to say "
        "to (a bug I hit, a workaround, a question) over ones I don't."
    )
    print(f"\n{briefing}")


if __name__ == "__main__":
    main()
