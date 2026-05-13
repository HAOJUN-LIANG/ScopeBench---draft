"""Main entry point for the CulturalScopeBench evaluation pipeline.

Usage examples:
    python run.py --task 1 --model gpt-4.1
    python run.py --task 2a --model all --resume
    python run.py --task 3 --stage 1
    python run.py --task 3 --stage 2 --model claude-sonnet-4.6
    python run.py --task 3 --stage 3 --model all
    python run.py --task all --model all --resume
    python run.py --task 1 --model gpt-4.1 --metrics-only
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_client import list_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASKS   = ["1", "2a", "2b", "3"]
RESULTS = Path(__file__).parent / "results"


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list_models()
    return [model_arg]


def run_task(task: str, model_name: str, stage: int | None, resume: bool) -> None:
    if task == "1":
        from tasks.task1 import run_task1
        from metrics.compute_sir_sdr_spr import compute_task1, save_metrics
        run_task1(model_name, resume=resume)
        save_metrics(compute_task1(model_name), "task1", model_name)

    elif task == "2a":
        from tasks.task2a import run_task2a
        from metrics.compute_sir_sdr_spr import compute_task2a, save_metrics
        run_task2a(model_name, resume=resume)
        save_metrics(compute_task2a(model_name), "task2a", model_name)

    elif task == "2b":
        from tasks.task2b import run_task2b
        from metrics.compute_sir_sdr_spr import compute_task2b, save_metrics
        run_task2b(model_name, resume=resume)
        save_metrics(compute_task2b(model_name), "task2b", model_name)

    elif task == "3":
        if stage == 1:
            from tasks.task3 import run_stage1
            run_stage1(resume=resume)

        elif stage == 2:
            from tasks.task3 import run_stage2
            run_stage2(model_name, resume=resume)

        elif stage == 3:
            from tasks.task3 import run_stage3
            from metrics.compute_hr import compute_task3, save_metrics
            run_stage3(model_name, resume=resume)
            save_metrics(compute_task3(model_name), model_name)

        else:
            # Run all three stages in order
            from tasks.task3 import run_stage1, run_stage2, run_stage3
            from metrics.compute_hr import compute_task3, save_metrics
            run_stage1(resume=resume)
            run_stage2(model_name, resume=resume)
            run_stage3(model_name, resume=resume)
            save_metrics(compute_task3(model_name), model_name)


def metrics_only(task: str, model_name: str) -> None:
    if task == "1":
        from metrics.compute_sir_sdr_spr import compute_task1, save_metrics
        save_metrics(compute_task1(model_name), "task1", model_name)
    elif task == "2a":
        from metrics.compute_sir_sdr_spr import compute_task2a, save_metrics
        save_metrics(compute_task2a(model_name), "task2a", model_name)
    elif task == "2b":
        from metrics.compute_sir_sdr_spr import compute_task2b, save_metrics
        save_metrics(compute_task2b(model_name), "task2b", model_name)
    elif task == "3":
        from metrics.compute_hr import compute_task3, save_metrics
        save_metrics(compute_task3(model_name), model_name)


def compile_summary() -> None:
    """Aggregate all per-model result JSONs into results/summary.csv."""
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        data = json.loads(f.read_text())
        rows.append(data)

    if not rows:
        logger.warning("No result files found in results/.")
        return

    all_keys = sorted({k for r in rows for k in r.keys()})
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary written to {out} ({len(rows)} rows).")


def main() -> None:
    parser = argparse.ArgumentParser(description="CulturalScopeBench evaluation pipeline")
    parser.add_argument("--task",   required=True, choices=TASKS + ["all"])
    parser.add_argument("--model",  default="all", help="Model name or 'all'")
    parser.add_argument("--stage",  type=int, choices=[1, 2, 3], default=None,
                        help="Task 3 only: run a specific stage (1/2/3)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed items (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics from existing outputs without calling APIs")
    parser.add_argument("--summary", action="store_true",
                        help="Compile results/summary.csv from all result JSONs and exit")
    args = parser.parse_args()

    if args.summary:
        compile_summary()
        return

    tasks   = TASKS if args.task == "all" else [args.task]
    # Stage 1 of Task 3 doesn't require a model; skip model loop for it
    for task in tasks:
        if task == "3" and args.stage == 1:
            if args.metrics_only:
                logger.info("Stage 1 generates statements; no metrics to compute.")
            else:
                run_task("3", "", stage=1, resume=args.resume)
            continue

        models = resolve_models(args.model)
        for model_name in models:
            logger.info(f"▶ task={task}  model={model_name}  stage={args.stage}")
            if args.metrics_only:
                metrics_only(task, model_name)
            else:
                run_task(task, model_name, stage=args.stage, resume=args.resume)

    compile_summary()


if __name__ == "__main__":
    main()
