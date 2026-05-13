"""Task 2b: Per-Option Independent Judgment (async, rate-limited)."""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_norms
from distractor_sampler import build_candidates, build_candidates_universal
from data_loader import get_cluster_map
from model_client import call_model
from prompt_builder import build_task2b_prompt

logger = logging.getLogger(__name__)

OUTPUT_ROOT    = Path(__file__).parent.parent.parent / "outputs" / "task2b"
CANDIDATE_FILE = (Path(__file__).parent.parent.parent / "outputs" / "task2a" / "candidates.json")

# Max concurrent API calls; tune to stay within rate limits
CONCURRENCY = int(os.environ.get("T2B_CONCURRENCY", "5"))


def _load_candidates() -> dict[int, dict]:
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            "Run Task 2a candidate generation first (or run Task 2a with any model)."
        )
    raw = json.loads(CANDIDATE_FILE.read_text())
    return {int(k): v for k, v in raw.items()}


def _parse_answer(text: str) -> str:
    match = re.search(r"Answer\s*:\s*(Yes|No)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    upper = text.upper()
    if upper.strip().startswith("YES"):
        return "Yes"
    if upper.strip().startswith("NO"):
        return "No"
    return "UNKNOWN"


async def _call_async(
    semaphore: asyncio.Semaphore,
    model_name: str,
    norm: dict,
    country: str,
) -> dict:
    async with semaphore:
        loop    = asyncio.get_event_loop()
        prompt  = build_task2b_prompt(norm, country)
        raw     = await loop.run_in_executor(
            None, lambda: call_model(model_name, prompt, temperature=0.0)
        )
        return {
            "global_id": int(norm["global_id"]),
            "scope":     norm["scope"],
            "country":   country,
            "answer":    _parse_answer(raw),
            "raw_response": raw,
        }


async def _run_async(model_name: str, todo_calls: list[tuple[dict, str]]) -> list[dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        _call_async(semaphore, model_name, norm, country)
        for norm, country in todo_calls
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


def run_task2b(model_name: str, resume: bool = True) -> Path:
    out_dir  = OUTPUT_ROOT / model_name
    out_file = out_dir / "responses.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple] = set()
    results: list[dict] = []
    if resume and out_file.exists():
        results = json.loads(out_file.read_text())
        done_keys = {(r["global_id"], r["country"]) for r in results}
        logger.info(f"[Task2b/{model_name}] Resuming — {len(done_keys)} calls already done.")

    norms      = load_norms()
    candidates = _load_candidates()
    cluster_map = get_cluster_map()

    # Build (norm, country) pairs for all 4 candidates per specific norm
    todo_calls: list[tuple[dict, str]] = []
    for norm in norms:
        gid   = int(norm["global_id"])
        cands = candidates.get(gid)
        if cands is None:
            continue
        if norm["scope"] == "Specific":
            countries = cands["options"]  # 4 country names (A–D)
        else:
            # Universal norms: use a sample of 4 random countries for robustness
            continue  # skip Universal norms in Task 2b

        for country in countries:
            if (gid, country) not in done_keys:
                todo_calls.append((norm, country))

    logger.info(f"[Task2b/{model_name}] {len(todo_calls)} calls remaining.")

    new_results = asyncio.run(_run_async(model_name, todo_calls))
    results.extend(new_results)

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Task2b/{model_name}] Complete. Saved to {out_file}.")
    return out_file
