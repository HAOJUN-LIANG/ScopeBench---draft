"""Task 3: Behavioral Safety — three-stage pipeline."""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_norms, get_cluster_map
from distractor_sampler import get_wrong_country
from model_client import call_model, get_judge_model_name
from prompt_builder import (
    build_task3_generation_prompt,
    build_task3_eval_prompt,
    build_task3_judge_prompt,
)

logger = logging.getLogger(__name__)

OUTPUT_ROOT     = Path(__file__).parent.parent.parent / "outputs" / "task3"
STATEMENTS_FILE = OUTPUT_ROOT / "statements.json"
CONCURRENCY     = int(os.environ.get("T3_CONCURRENCY", "5"))


# ── Stage 1: Statement generation (judge model, run once) ──────────────────

def run_stage1(resume: bool = True) -> Path:
    """Generate scope-violating statements for all norms using the judge model."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    results: list[dict] = []
    if resume and STATEMENTS_FILE.exists():
        results = json.loads(STATEMENTS_FILE.read_text())
        done_ids = {r["norm_id"] for r in results}
        logger.info(f"[Task3/Stage1] Resuming — {len(done_ids)} norms already done.")

    norms       = load_norms()
    cluster_map = get_cluster_map()
    judge       = get_judge_model_name()

    todo = [n for n in norms if n["global_id"] not in done_ids]
    logger.info(f"[Task3/Stage1] {len(todo)} norms remaining.")

    for i, norm in enumerate(todo):
        scope      = norm["scope"]
        statements = {}

        if scope == "Universal":
            # SDR: universal → presented as culture-specific
            raw = call_model(judge, build_task3_generation_prompt(norm, "SDR"), temperature=0.7)
            statements["SDR"] = raw.strip()

        elif scope == "Specific":
            # SIR: specific → presented as universal
            raw_sir = call_model(judge, build_task3_generation_prompt(norm, "SIR"), temperature=0.7)
            statements["SIR"] = raw_sir.strip()

            # SPR: specific → misattributed to wrong country
            wrong = get_wrong_country(norm, cluster_map)
            norm_with_wrong = {**norm, "wrong_country": wrong}
            raw_spr = call_model(judge, build_task3_generation_prompt(norm_with_wrong, "SPR"), temperature=0.7)
            statements["SPR"] = raw_spr.strip()
            statements["wrong_country"] = wrong

        results.append({
            "norm_id":    norm["global_id"],
            "scope":      scope,
            "rule":       norm["rule"],
            "country":    norm["country/source"],
            "statements": statements,
        })

        if (i + 1) % 50 == 0:
            STATEMENTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            logger.info(f"[Task3/Stage1] {i + 1}/{len(todo)} done (checkpoint).")

    STATEMENTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Task3/Stage1] Complete. Saved to {STATEMENTS_FILE}.")
    return STATEMENTS_FILE


# ── Stage 2: Model evaluation (24 eval models) ─────────────────────────────

def _load_statements() -> list[dict]:
    if not STATEMENTS_FILE.exists():
        raise FileNotFoundError("Run Stage 1 first to generate statements.")
    return json.loads(STATEMENTS_FILE.read_text())


def _iter_statement_items(statements: list[dict]):
    """Yield (norm_id, error_type, statement_text) for every statement."""
    for entry in statements:
        for etype, text in entry["statements"].items():
            if etype == "wrong_country":
                continue
            yield entry["norm_id"], etype, text


async def _eval_async(
    semaphore: asyncio.Semaphore,
    model_name: str,
    norm_id: str,
    error_type: str,
    statement: str,
) -> dict:
    async with semaphore:
        loop   = asyncio.get_event_loop()
        prompt = build_task3_eval_prompt(statement)
        raw    = await loop.run_in_executor(
            None, lambda: call_model(model_name, prompt, temperature=0.0)
        )
        return {
            "norm_id":      norm_id,
            "error_type":   error_type,
            "statement":    statement,
            "raw_response": raw,
        }


def run_stage2(model_name: str, resume: bool = True) -> Path:
    out_dir  = OUTPUT_ROOT / model_name
    out_file = out_dir / "responses.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple] = set()
    results: list[dict] = []
    if resume and out_file.exists():
        results = json.loads(out_file.read_text())
        done_keys = {(r["norm_id"], r["error_type"]) for r in results}
        logger.info(f"[Task3/Stage2/{model_name}] Resuming — {len(done_keys)} done.")

    statements = _load_statements()
    todo = [
        (nid, etype, text)
        for nid, etype, text in _iter_statement_items(statements)
        if (nid, etype) not in done_keys
    ]
    logger.info(f"[Task3/Stage2/{model_name}] {len(todo)} calls remaining.")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def _run():
        tasks = [
            _eval_async(semaphore, model_name, nid, etype, text)
            for nid, etype, text in todo
        ]
        batch = []
        for coro in asyncio.as_completed(tasks):
            batch.append(await coro)
        return batch

    new_results = asyncio.run(_run())
    results.extend(new_results)
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Task3/Stage2/{model_name}] Complete. Saved to {out_file}.")
    return out_file


# ── Stage 3: Judge (judge model) ───────────────────────────────────────────

def _parse_judgment(text: str) -> str:
    text = text.strip()
    for label in ("Appropriate", "Evasive", "Harmful"):
        if label.lower() in text.lower():
            return label
    return "UNKNOWN"


def run_stage3(model_name: str, resume: bool = True) -> Path:
    responses_file = OUTPUT_ROOT / model_name / "responses.json"
    if not responses_file.exists():
        raise FileNotFoundError(f"Run Stage 2 for {model_name} first.")

    out_file = OUTPUT_ROOT / model_name / "judgments.json"
    judge    = get_judge_model_name()

    done_keys: set[tuple] = set()
    judgments: list[dict] = []
    if resume and out_file.exists():
        judgments = json.loads(out_file.read_text())
        done_keys = {(j["norm_id"], j["error_type"]) for j in judgments}
        logger.info(f"[Task3/Stage3/{model_name}] Resuming — {len(done_keys)} done.")

    responses = json.loads(responses_file.read_text())
    todo = [r for r in responses if (r["norm_id"], r["error_type"]) not in done_keys]
    logger.info(f"[Task3/Stage3/{model_name}] {len(todo)} calls remaining.")

    for i, resp in enumerate(todo):
        prompt   = build_task3_judge_prompt(
            resp["statement"], resp["raw_response"], resp["error_type"]
        )
        raw      = call_model(judge, prompt, temperature=0.0)
        judgment = _parse_judgment(raw)

        judgments.append({
            "norm_id":      resp["norm_id"],
            "error_type":   resp["error_type"],
            "judgment":     judgment,
            "raw_judgment": raw,
        })

        if (i + 1) % 100 == 0:
            out_file.write_text(json.dumps(judgments, ensure_ascii=False, indent=2))
            logger.info(f"[Task3/Stage3/{model_name}] {i + 1}/{len(todo)} done (checkpoint).")

    out_file.write_text(json.dumps(judgments, ensure_ascii=False, indent=2))
    logger.info(f"[Task3/Stage3/{model_name}] Complete. Saved to {out_file}.")
    return out_file
