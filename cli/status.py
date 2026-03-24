"""Status command — summarize tracked experiments, SLURM jobs, and open tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def _render_status_report(
    *,
    recipe_id: str | None,
    experiments: list[dict],
    tasks: list[dict],
    slurm_jobs: list[dict] | None = None,
) -> str:
    lines = ["# Auto-Coder-Trainer Status", ""]
    if recipe_id:
        lines.append(f"- **Recipe Filter**: {recipe_id}")
    lines.append(f"- **Tracked Experiments**: {len(experiments)}")
    lines.append(f"- **Visible Tasks**: {len(tasks)}")
    if slurm_jobs is not None:
        lines.append(f"- **Tracked SLURM Jobs**: {len(slurm_jobs)}")
    lines.append("")

    # SLURM jobs section
    if slurm_jobs:
        lines.extend(
            [
                "## SLURM Jobs",
                "| Job ID | Experiment | Stage | Status | Elapsed | Submitted |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for job in slurm_jobs:
            lines.append(
                "| {job_id} | {experiment_id} | {stage} | {status} | {elapsed} | {submitted} |".format(
                    job_id=job.get("job_id", "?"),
                    experiment_id=job.get("experiment_id", "?"),
                    stage=job.get("stage", "?"),
                    status=job.get("status", "?"),
                    elapsed=job.get("elapsed") or "-",
                    submitted=job.get("submitted_at", "?"),
                )
            )
        lines.append("")

    if tasks:
        lines.extend(
            [
                "## Tasks",
                "| ID | Recipe | Experiment | Status | Priority | Kind | Title |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in tasks:
            lines.append(
                "| {id} | {recipe_id} | {experiment_id} | {status} | {priority} | {kind} | {title} |".format(
                    id=task.get("id", "?"),
                    recipe_id=task.get("recipe_id", "?"),
                    experiment_id=task.get("experiment_id") or "n/a",
                    status=task.get("status", "?"),
                    priority=task.get("priority", "?"),
                    kind=task.get("kind", "?"),
                    title=task.get("title", "?"),
                )
            )
        lines.append("")

    if experiments:
        lines.extend(
            [
                "## Experiments",
                "| ID | Recipe | Status | Trainer | Backend | Metrics |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for experiment in experiments:
            metrics = experiment.get("metrics_json", {})
            metric_text = ", ".join(
                f"{key}={value}"
                for key, value in sorted(metrics.items())
                if isinstance(value, (int, float))
            ) if isinstance(metrics, dict) else ""
            lines.append(
                "| {id} | {recipe_id} | {status} | {trainer_type} | {backend} | {metrics} |".format(
                    id=experiment.get("id", "?"),
                    recipe_id=experiment.get("recipe_id", "?"),
                    status=experiment.get("status", "?"),
                    trainer_type=experiment.get("trainer_type", "?"),
                    backend=experiment.get("backend", "?"),
                    metrics=metric_text or "-",
                )
            )
        lines.append("")

    if not tasks and not experiments and not slurm_jobs:
        lines.append("_No tracked experiments or tasks yet._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _poll_slurm_live(slurm_jobs: list[dict], db) -> list[dict]:
    """Poll sacct for live status of non-terminal SLURM jobs and update DB.

    Returns the updated job list. Fails silently if sacct is unavailable.
    """
    terminal = frozenset({
        "COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT",
        "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
    })
    try:
        from trainers.slurm.submitter import check_job_status
    except ImportError:
        return slurm_jobs

    updated = []
    for job in slurm_jobs:
        if job.get("status", "") in terminal:
            updated.append(job)
            continue
        try:
            live = check_job_status(job["job_id"])
            job["status"] = live.get("state", job.get("status", "?"))
            job["elapsed"] = live.get("elapsed", job.get("elapsed"))
            job["exit_code"] = live.get("exit_code", job.get("exit_code"))
            if job["status"] in terminal and not job.get("finished_at"):
                from datetime import datetime, timezone
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
            db.update_slurm_job_status(
                job["job_id"],
                job["status"],
                elapsed=job.get("elapsed"),
                exit_code=job.get("exit_code"),
                finished_at=job.get("finished_at"),
            )
        except Exception:
            pass  # sacct not available or job not found
        updated.append(job)
    return updated


def run_status(args: argparse.Namespace) -> None:
    """Print a project-wide status summary."""
    from results.db import ResultDB

    recipe_id = getattr(args, "recipe_id", None)
    open_only = getattr(args, "open_only", False)
    output = getattr(args, "output", None)
    show_slurm = getattr(args, "slurm", False)

    db = ResultDB()
    try:
        db.connect()
    except Exception as exc:
        print(f"[status] Error connecting to results DB: {exc}")
        return

    try:
        experiments = db.list_experiments(recipe_id=recipe_id, limit=None)
        tasks = db.get_open_tasks(recipe_id=recipe_id) if open_only else db.get_tasks(recipe_id=recipe_id)

        slurm_jobs = None
        if show_slurm:
            slurm_jobs = db.get_slurm_jobs(recipe_id=recipe_id) if recipe_id else db.get_active_slurm_jobs()
            slurm_jobs = _poll_slurm_live(slurm_jobs, db)

        report = _render_status_report(
            recipe_id=recipe_id,
            experiments=experiments,
            tasks=tasks,
            slurm_jobs=slurm_jobs,
        )
    finally:
        db.close()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"[status] Report written to {output_path}")

    print(report)
