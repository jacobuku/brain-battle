"""Build the personal brain from the files in data/. Run once before followup.py.

    uv run python ingest.py

Each file goes to cognee.remember(), which extracts entities and relationships with an
LLM and adds them to one knowledge graph. Writes brain.html when done.

Follows the pattern from https://github.com/sandhya-subramani/Agent-with-a-Brain
(ingest.py / config.py), adapted to use Anthropic Claude Sonnet via cognee's
LLM_PROVIDER=custom + LiteLLM workaround instead of Bedrock (see .env).
"""

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# cognee reads its LLM key from LLM_API_KEY; reuse ANTHROPIC_API_KEY at runtime
# instead of duplicating the secret into .env or any file.
os.environ.setdefault("LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])

# Keep cognee's local graph + vector DBs inside this project folder instead of
# inside site-packages.
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(ROOT / ".data_storage"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(ROOT / ".cognee_system"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(ROOT / ".cognee_cache"))

import cognee  # noqa: E402  (must import after the env vars above are set)

DATA = ROOT / "data"


async def main() -> None:
    files = sorted(p for p in DATA.iterdir() if p.is_file() and not p.name.startswith("."))

    print("Clearing any previous graph...")
    await cognee.forget(everything=True)

    print(f"Building the personal brain from {len(files)} files:\n")
    for path in files:
        t = time.time()
        await cognee.remember(f"Source file: {path.name}\n\n{path.read_text()}", self_improvement=False)
        print(f"  remembered {path.name:<22} {time.time() - t:5.1f}s")

    html = ROOT / "brain.html"
    await cognee.visualize_graph(str(html))
    print(f"\nDone. Open the graph:  open {html}")


if __name__ == "__main__":
    asyncio.run(main())
