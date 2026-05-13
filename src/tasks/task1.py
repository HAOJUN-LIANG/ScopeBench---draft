"""Task 1: Norm Scope Classification."""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_norms
from model_client import call_model
from prompt_builder import build_task1_prompt

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).parent.parent.parent / "outputs" / "task1"


def _parse_label(text: str) -> str:
    """Extract UNIVERSAL or SPECIFIC from model output."""
    match = re.search(r"Label\s*:\s*(UNIVERSAL|SPECIFIC)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # fallback: scan for keyword
    upper = text.upper()
    if "UNIVERSAL" in upper and "SPECIFIC" not in upper:
        return "UNIVERSAL"
    if "SPECIFIC" in upper and "UNIVERSAL" not in upper:
        return "SPECIFIC"
    return "UNKNOWN"


def run_task1(model_name: str, resume: bool = True) -> Path:
    out_dir  = OUTPUT_ROOT / model_name
    out_file = out_dir / "responses.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint
    done_ids: set[int] = set()
    results: list[dict] = []
    if resume and out_file.exists():
        results = json.loads(out_file.read_text())
        done_ids = {r["global_id"] for r in results}
        logger.info(f"[Task1/{model_name}] Resuming — {len(done_ids)} already done.")

    norms = load_norms()
    todo  = [n for n in norms if int(n["global_id"]) not in done_ids]
    logger.info(f"[Task1/{model_name}] {len(todo)} norms remaining.")

    for i, norm in enumerate(todo):
        prompt   = build_task1_prompt(norm)
        raw      = call_model(model_name, prompt, temperature=0.0)
        label    = _parse_label(raw)

        results.append({
            "global_id":    int(norm["global_id"]),
            "scope":        norm["scope"],
            "predicted":    label,
            "raw_response": raw,
        })

        if (i + 1) % 50 == 0:
            out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            logger.info(f"[Task1/{model_name}] {i + 1}/{len(todo)} done (checkpoint saved).")

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Task1/{model_name}] Complete. Saved to {out_file}.")
    return out_file
