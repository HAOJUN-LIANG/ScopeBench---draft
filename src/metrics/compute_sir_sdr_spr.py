"""Compute SIR, SDR, SPR metrics for Task 1, 2a, 2b."""

import json
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path(__file__).parent.parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _safe_div(num: int, den: int) -> float:
    return round(num / den, 4) if den > 0 else 0.0


# ── Task 1 ────────────────────────────────────────────────────────────────────

def compute_task1(model_name: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / "task1" / model_name / "responses.json"
    data = json.loads(path.read_text())

    specific = [r for r in data if r["scope"] == "Specific"]
    universal = [r for r in data if r["scope"] == "Universal"]

    # SIR: SPECIFIC norm predicted as UNIVERSAL
    sir_num = sum(1 for r in specific if r["predicted"] == "UNIVERSAL")
    # SDR: UNIVERSAL norm predicted as SPECIFIC
    sdr_num = sum(1 for r in universal if r["predicted"] == "SPECIFIC")

    return {
        "task": "task1",
        "model": model_name,
        "n_specific": len(specific),
        "n_universal": len(universal),
        "SIR": _safe_div(sir_num, len(specific)),
        "SDR": _safe_div(sdr_num, len(universal)),
        "unknown_rate": _safe_div(
            sum(1 for r in data if r["predicted"] == "UNKNOWN"), len(data)
        ),
    }


# ── Task 2a ───────────────────────────────────────────────────────────────────

def compute_task2a(model_name: str) -> dict[str, Any]:
    resp_path = OUTPUT_ROOT / "task2a" / model_name / "responses.json"
    cand_path = OUTPUT_ROOT / "task2a" / "candidates.json"
    data      = json.loads(resp_path.read_text())
    cands_raw = json.loads(cand_path.read_text())
    cands     = {int(k): v for k, v in cands_raw.items()}

    specific  = [r for r in data if r["scope"] == "Specific"]
    universal = [r for r in data if r["scope"] == "Universal"]

    # SIR: SPECIFIC predicted as E (universal)
    sir_num = sum(1 for r in specific if r["predicted"] == "E")
    # SDR: UNIVERSAL not predicted as E
    sdr_num = sum(1 for r in universal if r["predicted"] != "E")

    # SPR: gold_letter not chosen and not E (predicted wrong country)
    spr_total, spr_hard, spr_easy = 0, 0, 0
    for r in specific:
        pred = r["predicted"]
        gid  = r["global_id"]
        c    = cands.get(gid, {})
        if pred in ("E", "F", "UNKNOWN"):
            continue
        # Find which country the model predicted
        letters = ["A", "B", "C", "D"]
        try:
            pred_country = r["options"][letters.index(pred)]
        except (ValueError, IndexError):
            continue

        gold    = c.get("gold")
        hard    = c.get("hard_distractor")
        easy    = c.get("easy_distractors", [])

        if pred_country != gold:
            spr_total += 1
            if pred_country == hard:
                spr_hard += 1
            elif pred_country in easy:
                spr_easy += 1

    n_specific = len(specific)
    return {
        "task": "task2a",
        "model": model_name,
        "n_specific": n_specific,
        "n_universal": len(universal),
        "SIR":      _safe_div(sir_num, n_specific),
        "SDR":      _safe_div(sdr_num, len(universal)),
        "SPR":      _safe_div(spr_total, n_specific),
        "SPR_hard": _safe_div(spr_hard, n_specific),
        "SPR_easy": _safe_div(spr_easy, n_specific),
        "CPG":      round(_safe_div(spr_hard, n_specific) - _safe_div(spr_easy, n_specific), 4),
    }


# ── Task 2b ───────────────────────────────────────────────────────────────────

def compute_task2b(model_name: str) -> dict[str, Any]:
    resp_path = OUTPUT_ROOT / "task2b" / model_name / "responses.json"
    cand_path = OUTPUT_ROOT / "task2a" / "candidates.json"
    data      = json.loads(resp_path.read_text())
    cands_raw = json.loads(cand_path.read_text())
    cands     = {int(k): v for k, v in cands_raw.items()}

    # Group responses by norm_id
    by_norm: dict[int, list[dict]] = {}
    for r in data:
        by_norm.setdefault(r["global_id"], []).append(r)

    sir_num = sdr_num = spr_total = spr_hard = spr_easy = 0
    n_specific = n_universal = 0

    for gid, responses in by_norm.items():
        c     = cands.get(gid, {})
        scope = responses[0]["scope"]
        gold  = c.get("gold")
        hard  = c.get("hard_distractor")
        easy  = c.get("easy_distractors", [])

        yes_countries = [r["country"] for r in responses if r["answer"] == "Yes"]

        if scope == "Specific":
            n_specific += 1
            non_gold_yes = [ctry for ctry in yes_countries if ctry != gold]
            # SIR: ≥2 non-source countries judged Yes
            if len(non_gold_yes) >= 2:
                sir_num += 1
            # SPR: source country NOT judged Yes, or any distractor judged Yes
            gold_judged_yes = gold in yes_countries
            if not gold_judged_yes:
                spr_total += 1
                if hard in yes_countries:
                    spr_hard += 1
                if any(e in yes_countries for e in easy):
                    spr_easy += 1

    return {
        "task": "task2b",
        "model": model_name,
        "n_specific": n_specific,
        "SIR":      _safe_div(sir_num, n_specific),
        "SPR":      _safe_div(spr_total, n_specific),
        "SPR_hard": _safe_div(spr_hard, n_specific),
        "SPR_easy": _safe_div(spr_easy, n_specific),
        "CPG":      round(_safe_div(spr_hard, n_specific) - _safe_div(spr_easy, n_specific), 4),
    }


# ── Save results ──────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, task: str, model_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{task}_{model_name}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
