import csv
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "norms.csv"


def load_norms(
    scope: Optional[str] = None,
    norm_type: Optional[str] = None,
    cluster: Optional[str] = None,
) -> list[dict]:
    """Load norms from norms.csv with optional filters.

    Args:
        scope: 'Universal', 'Specific', or None for all.
        norm_type: filter by norm_type field.
        cluster: filter by cultural_cluster field.
    """
    norms = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if scope and row["scope"] != scope:
                continue
            if norm_type and row["norm_type"] != norm_type:
                continue
            if cluster and row["cultural_cluster"] != cluster:
                continue
            norms.append(row)
    return norms


def load_norm_by_id(global_id: int) -> Optional[dict]:
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["global_id"]) == global_id:
                return row
    return None


def get_all_countries() -> list[str]:
    countries = set()
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "Specific":
                countries.add(row["country/source"])
    return sorted(countries)


def get_cluster_map() -> dict[str, list[str]]:
    """Returns {cluster_name: [country, ...]} for Specific norms only."""
    mapping: dict[str, set[str]] = {}
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "Specific":
                continue
            cluster = row["cultural_cluster"]
            country = row["country/source"]
            mapping.setdefault(cluster, set()).add(country)
    return {k: sorted(v) for k, v in mapping.items()}
