"""Task 2a: Multiple-Choice Attribution."""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_norms, get_cluster_map
from distractor_sampler import build_candidates, build_candidates_universal
from model_client import call_model
from prompt_builder import build_task2a_prompt

logger = logging.getLogger(__name__)

OUTPUT_ROOT    = Path(__file__).parent.parent.parent / "outputs" / "task2a"
CANDIDATE_FILE = OUTPUT_ROOT / "candidates.json"


def _build_or_load_candidates(norms: list[dict]) -> dict[int, dict]:
    """Pre-generate candidates for all norms once and cache to disk."""
    if CANDIDATE_FILE.exists():
        raw = json.loads(CANDIDATE_FILE.read_text())
        return {int(k): v for k, v in raw.items()}

    cluster_map = get_cluster_map()
    candidates: dict[int, dict] = {}
    for norm in norms:
        gid = int(norm["global_id"])
        if norm["scope"] == "Specific":
            candidates[gid] = build_candidates(norm, cluster_map)
        else:
            candidates[gid] = build_candidates_universal()

    CANDIDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_FILE.write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    logger.info(f"Candidates saved to {CANDIDATE_FILE}.")
    return candidates


def _parse_letter(text: str) -> str:
    match = re.search(r"\b([A-Fa-f])\b", text.strip())
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def run_task2a(model_name: str, resume: bool = True) -> Path:
    out_dir  = OUTPUT_ROOT / model_name
    out_file = out_dir / "responses.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_ids: set[int] = set()
    results: list[dict] = []
    if resume and out_file.exists():
        results = json.loads(out_file.read_text())
        done_ids = {r["global_id"] for r in results}
        logger.info(f"[Task2a/{model_name}] Resuming — {len(done_ids)} already done.")

    norms      = load_norms()
    candidates = _build_or_load_candidates(norms)
    todo       = [n for n in norms if int(n["global_id"]) not in done_ids]
    logger.info(f"[Task2a/{model_name}] {len(todo)} norms remaining.")

    for i, norm in enumerate(todo):
        gid   = int(norm["global_id"])
        cands = candidates[gid]
        prompt = build_task2a_prompt(norm, cands)
        raw    = call_model(model_name, prompt, temperature=0.0)
        letter = _parse_letter(raw)

        results.append({
            "global_id":    gid,
            "scope":        norm["scope"],
            "gold_letter":  cands["gold_letter"],
            "gold_country": cands["gold"],
            "options":      cands["options"],
            "predicted":    letter,
            "raw_response": raw,
        })

        if (i + 1) % 50 == 0:
            out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            logger.info(f"[Task2a/{model_name}] {i + 1}/{len(todo)} done.")

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Task2a/{model_name}] Complete. Saved to {out_file}.")
    return out_file
