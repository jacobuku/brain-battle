# brain-battle

A personal brain that tracks your follow-up debt.

Built for the [Battle of the Personal Brains Hackathon](https://luma.com/sep-21-cognee-aws)
(Cognee + AWS Strands Agents + Docker Sandboxes + Bright Data).

## What it does

Three stages, one run:

1. **Before the event** -- scrapes the event page (via Bright Data) to find the
   host/sponsor companies, then combines that with what's already in memory
   to write a briefing: who to find tonight and what to talk to them about.
2. **After the event** -- reads your notes, extracts every promise you made
   to someone (structured output: person, company, what you promised, due
   date, context), and prints it as a table.
3. **Take action** -- drafts a short follow-up for the first open
   commitment, asks for your approval (y/n) before it "sends" anything,
   writes it to an outbox, records the commitment as complete in memory, and
   re-answers "What do I still owe people?" to confirm it dropped off.

Memory persists across runs and across separate script/agent instances: it's
a [Cognee](https://www.cognee.ai/) knowledge graph on disk, not conversation
history.

## Run it

```bash
./reset.sh && uv run python demo.py
```

`reset.sh` clears the outbox and rebuilds the graph from scratch
(`cognee.forget(everything=True)` + re-ingest `data/` + regenerate
`brain.html`), so the demo always starts from a clean, reproducible state.
`demo.py` then runs all three stages end to end, prompting for a y/n
approval before it "sends" anything.

**`data/sample_notes.txt` is SAMPLE DATA** -- fabricated phone notes for the
demo, not anyone's real conversations. `demo.py` prints which source it's
using (`Notes source: SAMPLE DATA` or `REAL NOTES`, based on whether
`data/real_notes.txt` exists) so it's never ambiguous which you're looking
at.

## Stack

- **[Strands Agents](https://strandsagents.com/)** -- the agent loop, tools,
  memory manager, and steering (human-approval-before-tool-call) plugin.
- **[Cognee](https://www.cognee.ai/)** -- long-term memory as a knowledge
  graph (`remember`/`recall`), bridged into Strands via a small
  `CogneeMemory` class satisfying its `search`/`add` memory-store interface.
- **[Bright Data](https://brightdata.com/)** -- Web Unlocker REST API for
  scraping the event page.
- **Anthropic Claude Sonnet** -- the model behind every agent here.

## Security design

The agent that reads the web (`fetch_page`, via Bright Data) and the agent
that can write to long-term memory (`add_memory`) are two separate `Agent`
instances -- never the same one:

- **No memory-write access while reading untrusted content.** Scraped page
  text is untrusted input. An agent that could both fetch arbitrary pages
  and write to memory would let a page with injected instructions plant
  fabricated "facts" into the graph that later runs treat as ground truth.
  The briefing agent's `MemoryManager` is built with `add_tool_config=False`
  -- it can search memory, never write it.
- **Domain allowlist on `fetch_page`.** The tool refuses any URL outside a
  fixed set of domains (the event site plus the sponsor/host companies
  found on it), so injected instructions in a fetched page can't redirect it
  into fetching an attacker-controlled URL as an exfiltration channel.
- **Human approval before every send.** `send_followup` never runs without
  an explicit y/n from the terminal, gated by a Strands steering handler
  that runs before the tool call -- no LLM in that decision. A rejection
  feeds the reason back to the model so it revises and retries.
- **No secrets in logs.** Recalled/remembered memory content is truncated
  before it's printed, so notes with personal details don't get dumped
  wholesale to stdout on every recall.

## Known issues

- **cognee 1.1.2's `LLM_PROVIDER=anthropic` adapter is broken.** It calls
  `instructor.patch()` without `provider=Provider.ANTHROPIC`, so `instructor`
  defaults to `Provider.OPENAI` and rejects `anthropic_tools` mode with
  `Mode Mode.ANTHROPIC_TOOLS is not registered for provider Provider.OPENAI`.
  **Workaround used here:** `LLM_PROVIDER=custom`, which routes through
  LiteLLM instead and works correctly with `LLM_MODEL=anthropic/claude-sonnet-5`.
- **This project's `strands-agents` version requires an `extraction`
  attribute on any custom `MemoryStore`.** It's not documented in the
  reference integration this was built from
  ([Agent-with-a-Brain](https://github.com/sandhya-subramani/Agent-with-a-Brain)),
  and omitting it raises `AttributeError: 'CogneeMemory' object has no
  attribute 'extraction'` from inside `MemoryManager.__init__`. Fixed here by
  setting `extraction = None` on `CogneeMemory` (this project runs its own
  explicit structured-output extraction instead of the framework's automatic
  one).

## Files

| File | Purpose |
|---|---|
| `demo.py` | The judge-facing run: all three stages in sequence |
| `reset.sh` | Clean state: clear outbox, rebuild the graph |
| `ingest.py` | Builds the graph from `data/`, renders `brain.html` |
| `followup.py` | Stage 2 + 3 standalone (commitment extraction, draft/approve/send) |
| `prebrief.py` | Stage 1 standalone (event research + briefing) |
| `brain_test.py` | Minimal proof that Cognee memory persists across separate agent instances |
| `data/sample_notes.txt` | Sample data -- see note above |

---

LJ


## Screenshots

![Screenshot 1](Screenshot%202026-09-21%20at%2020.05.15.png)

![Screenshot 2](Screenshot%202026-09-21%20at%2020.06.14.png)
