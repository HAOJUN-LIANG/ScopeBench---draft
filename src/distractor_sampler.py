"""Distractor sampling for Task 2a and 2b.

For each SPECIFIC norm, builds a candidate list of 4 countries:
  - 1 gold (source country)
  - 1 same-cluster distractor  (Hard)
  - 2 different-cluster distractors (Easy)

Candidate order is deterministic via hash(global_id) to avoid position bias.
Pacific Islands has only 2 countries, so same-cluster is always deterministic.
"""

import hashlib
import random
from data_loader import get_cluster_map


def _stable_rng(norm_id: int, salt: str = "") -> random.Random:
    seed = hashlib.md5(f"{norm_id}{salt}".encode()).hexdigest()
    return random.Random(int(seed, 16))


def build_candidates(norm: dict, cluster_map: dict[str, list[str]]) -> dict:
    """Return candidate dict for a SPECIFIC norm.

    Returns:
        {
          "gold": str,
          "hard_distractor": str,
          "easy_distractors": [str, str],
          "options": [str, str, str, str],   # shuffled A–D
          "gold_letter": str,                # which letter is gold
        }
    """
    assert norm["scope"] == "Specific", "build_candidates only applies to Specific norms"

    norm_id  = int(norm["global_id"])
    gold     = norm["country/source"]
    cluster  = norm["cultural_cluster"]
    rng      = _stable_rng(norm_id)

    # ── same-cluster distractor ──────────────────────────────────
    same_cluster_pool = [c for c in cluster_map.get(cluster, []) if c != gold]
    if not same_cluster_pool:
        raise ValueError(f"No same-cluster candidates for {gold} in {cluster}")
    hard = rng.choice(same_cluster_pool)

    # ── different-cluster distractors ────────────────────────────
    other_countries = [
        c
        for cl, countries in cluster_map.items()
        if cl != cluster
        for c in countries
        if c != gold
    ]
    easy = rng.sample(other_countries, 2)

    # ── shuffle all 4 into A–D ───────────────────────────────────
    options = [gold, hard] + easy
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    gold_letter = letters[options.index(gold)]

    return {
        "gold": gold,
        "hard_distractor": hard,
        "easy_distractors": easy,
        "options": options,
        "gold_letter": gold_letter,
    }


def build_candidates_universal() -> dict:
    """For UNIVERSAL norms the gold answer is always E."""
    return {
        "gold": "Universal",
        "hard_distractor": None,
        "easy_distractors": [],
        "options": [],
        "gold_letter": "E",
    }


def get_wrong_country(norm: dict, cluster_map: dict[str, list[str]]) -> str:
    """Sample a wrong-attribution country for Task 3 SPR statement generation.

    Prefers a same-cluster country (more challenging misattribution).
    """
    norm_id = int(norm["global_id"])
    gold    = norm["country/source"]
    cluster = norm["cultural_cluster"]
    rng     = _stable_rng(norm_id, salt="spr")

    same_pool = [c for c in cluster_map.get(cluster, []) if c != gold]
    if same_pool:
        return rng.choice(same_pool)

    # fallback: any other country
    other = [
        c
        for cl, countries in cluster_map.items()
        if cl != cluster
        for c in countries
    ]
    return rng.choice(other)
