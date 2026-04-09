#!/usr/bin/env python3
"""
TinyZero Experiment Log Analyzer
=================================
Parses SLURM .out log files from TinyZero experiments and generates a detailed
Markdown report with per-experiment summaries, trend analysis, and cross-experiment
insights.

Usage:
    python3 analyze_experiments.py [LOG_DIR]
    python3 analyze_experiments.py /scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs

Output: Markdown report to stdout. Redirect to save:
    python3 analyze_experiments.py > report.md
"""

import os
import re
import sys
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = "/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs"

# Regex patterns for parsing log lines
# Step metrics line: step:15 - actor/kl_loss:0.011 - ...
STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")

# Individual metric within a step line: key:value
METRIC_RE = re.compile(r"([a-zA-Z0-9_/\-\.]+):([\-\d\.eE]+)")

# Sub-experiment start markers
GRPO_MARKER_RE = re.compile(r"^\s*GRPO:\s+(\S+)")
PPO_MARKER_RE = re.compile(r"^\s*PPO:\s+(\S+)")
SFT_MARKER_RE = re.compile(r"^\s*SFT:\s+(\S+)")
# Generic DONE marker: [DONE] exp01_grpo — Wed Apr  8 04:18:30 EDT 2026
DONE_MARKER_RE = re.compile(r"^\[DONE\]\s+(\S+)")

# Validation final score line:
# ('Final validation metrics: {\'val/test_score/openai/gsm8k\': np.float64(0.805)}')
FINAL_VAL_RE = re.compile(
    r"Final validation metrics:.*val/test_score/openai/gsm8k['\"]?\s*:\s*"
    r"(?:np\.float64\()?([\d\.]+)(?:\))?"
)

# Model info from GRPO/PPO header: Model: Qwen/Qwen2.5-3B  BS: 32  n: 4  ...
MODEL_INFO_RE = re.compile(r"Model:\s+(\S+)")

# Experiment failure patterns
OOM_RE = re.compile(r"(?:CUDA out of memory|OOM|OutOfMemoryError)", re.IGNORECASE)
TIMEOUT_RE = re.compile(r"(?:DUE TO TIME LIMIT|TIMEOUT|timed out|Cancelled)", re.IGNORECASE)
ERROR_RE = re.compile(r"(?:Traceback|Error|Exception|FAILED)", re.IGNORECASE)

# Log filename pattern: exptz-01-PPO-vs-GRPO_5549043.out  or  exptz-08-Rollout-N_5644914.out
LOG_FILENAME_RE = re.compile(r"exptz-(\d+)-(.+?)_(\d+)\.out")

# Known metric keys to extract per step
STEP_METRICS = [
    "val/test_score/openai/gsm8k",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/kl_loss",
    "perf/max_memory_allocated_gb",
    "timing_s/step",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class StepMetrics:
    """Metrics for a single training step."""

    def __init__(self, step: int):
        self.step = step
        self.data = {}

    def set(self, key: str, value: float):
        self.data[key] = value

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.data.get(key, default)


class SubExperiment:
    """A single sub-experiment run (e.g. exp08_n2)."""

    def __init__(self, name: str, algorithm: str = "unknown"):
        self.name = name
        self.algorithm = algorithm
        self.model = "unknown"
        self.steps = []  # List[StepMetrics]
        self.status = "ok"  # ok, oom, timeout, error, empty
        self.error_detail = ""
        self.final_validation_score = None

    def add_step(self, metrics: StepMetrics):
        self.steps.append(metrics)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def init_score(self) -> Optional[float]:
        if not self.steps:
            return None
        return self.steps[0].get("val/test_score/openai/gsm8k")

    @property
    def final_score(self) -> Optional[float]:
        if self.final_validation_score is not None:
            return self.final_validation_score
        if not self.steps:
            return None
        return self.steps[-1].get("val/test_score/openai/gsm8k")

    @property
    def delta_score(self) -> Optional[float]:
        i, f = self.init_score, self.final_score
        if i is None or f is None:
            return None
        return f - i

    def metric_series(self, key: str) -> list:
        """Return list of (step_number, value) for a metric."""
        result = []
        for s in self.steps:
            v = s.get(key)
            if v is not None:
                result.append((s.step, v))
        return result

    def metric_trend(self, key: str) -> str:
        """Describe the trend of a metric: 'up', 'down', 'stable', 'unstable', 'unknown'."""
        series = self.metric_series(key)
        if len(series) < 2:
            return "unknown"
        values = [v for _, v in series]
        # Compare first third vs last third
        n = len(values)
        head = values[: max(1, n // 3)]
        tail = values[-max(1, n // 3) :]
        avg_head = sum(head) / len(head)
        avg_tail = sum(tail) / len(tail)
        if avg_head == 0:
            return "unknown"
        pct_change = (avg_tail - avg_head) / abs(avg_head) * 100
        if pct_change > 15:
            return "up"
        elif pct_change < -15:
            return "down"
        else:
            # Check variance for stability
            if n >= 4:
                mean = sum(values) / n
                variance = sum((v - mean) ** 2 for v in values) / n
                std = math.sqrt(variance)
                cv = std / abs(mean) if mean != 0 else float("inf")
                if cv > 0.3:
                    return "unstable"
            return "stable"


class ExperimentFile:
    """A single SLURM log file, possibly containing multiple sub-experiments."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.exp_id = ""
        self.exp_name = ""
        self.slurm_job_id = ""
        self.sub_experiments = []  # List[SubExperiment]

        m = LOG_FILENAME_RE.match(self.filename)
        if m:
            self.exp_id = m.group(1)
            self.exp_name = m.group(2)
            self.slurm_job_id = m.group(3)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_metric_value(raw: str) -> Optional[float]:
    """Parse a metric value string to float. Returns None on failure."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def parse_step_line(line: str) -> Optional[StepMetrics]:
    """Parse a step metrics line into a StepMetrics object."""
    m = STEP_LINE_RE.search(line)
    if not m:
        return None
    step_num = int(m.group(1))
    metrics_str = m.group(2)

    sm = StepMetrics(step_num)
    # Split by ' - ' to get individual key:value pairs
    pairs = metrics_str.split(" - ")
    found_any = False
    for pair in pairs:
        pair = pair.strip()
        # Find the last colon to split key:value
        colon_idx = pair.rfind(":")
        if colon_idx > 0:
            key = pair[:colon_idx].strip()
            val_raw = pair[colon_idx + 1 :].strip()
            val = parse_metric_value(val_raw)
            if val is not None:
                sm.set(key, val)
                found_any = True

    return sm if found_any else None


def parse_log_file(filepath: str) -> ExperimentFile:
    """Parse a single .out log file into an ExperimentFile."""
    ef = ExperimentFile(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        sub = SubExperiment("PARSE_ERROR")
        sub.status = "error"
        sub.error_detail = str(e)
        ef.sub_experiments.append(sub)
        return ef

    if not lines:
        sub = SubExperiment("EMPTY")
        sub.status = "empty"
        ef.sub_experiments.append(sub)
        return ef

    current_sub = None
    seen_subs = set()

    for line in lines:
        stripped = line.strip()

        # Check for sub-experiment start markers
        sub_name = None
        algorithm = "unknown"

        m = GRPO_MARKER_RE.match(stripped)
        if m:
            sub_name = m.group(1)
            algorithm = "GRPO"
        else:
            m = PPO_MARKER_RE.match(stripped)
            if m:
                sub_name = m.group(1)
                algorithm = "PPO"
            else:
                m = SFT_MARKER_RE.match(stripped)
                if m:
                    sub_name = m.group(1)
                    algorithm = "SFT"

        if sub_name:
            # Start a new sub-experiment
            current_sub = SubExperiment(sub_name, algorithm)
            # Look for model info on the next few lines
            ef.sub_experiments.append(current_sub)
            seen_subs.add(sub_name)
            continue

        # Check for model info line (after sub-experiment marker)
        if current_sub and current_sub.model == "unknown":
            m = MODEL_INFO_RE.search(stripped)
            if m:
                current_sub.model = m.group(1)

        # Check for [DONE] marker
        m = DONE_MARKER_RE.match(stripped)
        if m:
            done_name = m.group(1)
            # Optionally finalize the sub-experiment
            continue

        # Check for final validation score
        m = FINAL_VAL_RE.search(stripped)
        if m and current_sub:
            val = parse_metric_value(m.group(1))
            if val is not None:
                current_sub.final_validation_score = val

        # Check for error patterns
        if current_sub and current_sub.status == "ok":
            if OOM_RE.search(stripped):
                current_sub.status = "oom"
                current_sub.error_detail = "CUDA Out of Memory"
            elif TIMEOUT_RE.search(stripped):
                current_sub.status = "timeout"
                current_sub.error_detail = "Time limit exceeded"
            elif ERROR_RE.search(stripped):
                # Only mark as error if it looks like a real failure, not just
                # a log line containing the word "error" in normal metrics
                if "Traceback" in stripped or "Exception" in stripped:
                    current_sub.status = "error"
                    # Capture next few lines as error detail
                    current_sub.error_detail = stripped[:200]

        # Parse step metrics
        step = parse_step_line(stripped)
        if step:
            if current_sub is None:
                # Metrics before any marker: create an unnamed sub-experiment
                current_sub = SubExperiment("unnamed", "unknown")
                ef.sub_experiments.append(current_sub)
            current_sub.add_step(step)

    # Post-processing: determine final status for sub-experiments with no steps
    for sub in ef.sub_experiments:
        if sub.status == "ok" and sub.total_steps == 0:
            sub.status = "empty"

    # If no sub-experiments found, create one from the whole file
    if not ef.sub_experiments:
        sub = SubExperiment("all", "unknown")
        sub.status = "empty"
        ef.sub_experiments.append(sub)

    return ef


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(val: Optional[float]) -> str:
    """Format a float as percentage string."""
    if val is None:
        return "-"
    return f"{val * 100:.1f}%"


def fmt_pct_change(val: Optional[float]) -> str:
    """Format a score delta as percentage points."""
    if val is None:
        return "-"
    return f"{val * 100:+.1f}pp"


def fmt_float(val: Optional[float], decimals: int = 3) -> str:
    """Format a float with given decimal places."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def fmt_trend_arrow(trend: str) -> str:
    """Convert trend string to arrow symbol."""
    return {
        "up": "up",
        "down": "down",
        "stable": "stable",
        "unstable": "volatile",
        "unknown": "?",
    }.get(trend, trend)


def score_bar(score: Optional[float], width: int = 20) -> str:
    """Generate a simple ASCII progress bar for a score 0-1."""
    if score is None:
        return " " * width
    filled = int(score * width)
    return "|" + "#" * filled + "-" * (width - filled) + "|"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_summary_table(experiments: list) -> str:
    """Generate the per-experiment summary table."""
    lines = []
    lines.append("## 1. Per-Experiment Summary")
    lines.append("")
    lines.append(
        "| Exp | Sub | Model | Algo | Steps | Init Score | Final Score | "
        "Delta | Resp Len (first->last) | Clip Ratio | Entropy | Grad Norm | "
        "Mem GB | Time/Step | Status |"
    )
    lines.append(
        "|------|------|-------|------|-------|------------|-------------|"
        "-------|------------------------|------------|---------|-----------|"
        "--------|-----------|--------|"
    )

    for ef in experiments:
        for sub in ef.sub_experiments:
            # Determine model display name
            model = sub.model
            if model.startswith("Qwen/Qwen2.5-"):
                model = model.replace("Qwen/Qwen2.5-", "Q2.5-")

            # Response length range
            rl_first = sub.steps[0].get("response_length/mean") if sub.steps else None
            rl_last = sub.steps[-1].get("response_length/mean") if sub.steps else None
            if rl_first is not None and rl_last is not None:
                rl_str = f"{rl_first:.0f}->{rl_last:.0f}"
            else:
                rl_str = "-"

            # Clip ratio (average of last 5 steps, or all)
            clip_vals = [v for _, v in sub.metric_series("response_length/clip_ratio")]
            clip_str = f"{sum(clip_vals) / len(clip_vals):.3f}" if clip_vals else "-"

            # Entropy (average)
            ent_vals = [v for _, v in sub.metric_series("actor/entropy_loss")]
            ent_str = f"{sum(ent_vals) / len(ent_vals):.3f}" if ent_vals else "-"

            # Grad norm (average)
            gn_vals = [v for _, v in sub.metric_series("actor/grad_norm")]
            gn_str = f"{sum(gn_vals) / len(gn_vals):.3f}" if gn_vals else "-"

            # Memory (peak)
            mem_vals = [v for _, v in sub.metric_series("perf/max_memory_allocated_gb")]
            mem_str = f"{max(mem_vals):.1f}" if mem_vals else "-"

            # Time per step (average)
            time_vals = [v for _, v in sub.metric_series("timing_s/step")]
            if time_vals:
                avg_time = sum(time_vals) / len(time_vals)
                if avg_time > 60:
                    time_str = f"{avg_time / 60:.1f}min"
                else:
                    time_str = f"{avg_time:.1f}s"
            else:
                time_str = "-"

            # Status display
            status_icon = {
                "ok": "OK",
                "oom": "OOM",
                "timeout": "TIMEOUT",
                "error": "ERROR",
                "empty": "EMPTY",
            }.get(sub.status, sub.status)

            lines.append(
                f"| {ef.exp_id} | {sub.name} | {model} | {sub.algorithm} | "
                f"{sub.total_steps} | {fmt_pct(sub.init_score)} | "
                f"{fmt_pct(sub.final_score)} | {fmt_pct_change(sub.delta_score)} | "
                f"{rl_str} | {clip_str} | {ent_str} | {gn_str} | "
                f"{mem_str} | {time_str} | {status_icon} |"
            )

    lines.append("")
    return "\n".join(lines)


def generate_detailed_analysis(experiments: list) -> str:
    """Generate per-experiment detailed analysis."""
    lines = []
    lines.append("## 2. Per-Experiment Detailed Analysis")
    lines.append("")

    for ef in experiments:
        lines.append(f"### Experiment {ef.exp_id}: {ef.exp_name.replace('-', ' ')}")
        lines.append(f"**File**: `{ef.filename}`  ")
        lines.append(f"**SLURM Job**: {ef.slurm_job_id}  ")
        lines.append("")

        for sub in ef.sub_experiments:
            lines.append(f"#### {sub.name} ({sub.algorithm}, {sub.model})")
            lines.append("")

            # Status
            if sub.status != "ok":
                lines.append(f"**Status**: {sub.status.upper()}")
                if sub.error_detail:
                    lines.append(f"  Detail: {sub.error_detail}")
                if sub.total_steps == 0:
                    lines.append("  No training steps recorded.")
                lines.append("")
                # Still print whatever metrics we have
                if sub.total_steps > 0:
                    lines.append("Partial metrics below:")
                    lines.append("")
                else:
                    continue

            # Score trajectory
            score_series = sub.metric_series("val/test_score/openai/gsm8k")
            if score_series:
                lines.append("**Score Trajectory**:")
                lines.append("```")
                for step, val in score_series:
                    bar = score_bar(val, width=30)
                    lines.append(f"  step {step:>3d}: {val:.4f} {bar} ({val * 100:.1f}%)")
                lines.append("```")
                trend = sub.metric_trend("val/test_score/openai/gsm8k")
                lines.append(f"Trend: {fmt_trend_arrow(trend)}")
            else:
                lines.append("**Score Trajectory**: No validation scores recorded.")
            lines.append("")

            # Response length
            rl_series = sub.metric_series("response_length/mean")
            rl_min_series = sub.metric_series("response_length/min")
            rl_max_series = sub.metric_series("response_length/max")
            if rl_series:
                rl_vals = [v for _, v in rl_series]
                rl_trend = sub.metric_trend("response_length/mean")
                rl_min = min(rl_vals)
                rl_max = max(rl_vals)
                rl_avg = sum(rl_vals) / len(rl_vals)
                lines.append(
                    f"**Response Length**: avg={rl_avg:.0f}, "
                    f"range=[{rl_min:.0f}, {rl_max:.0f}], trend={fmt_trend_arrow(rl_trend)}"
                )
                # Show min/max if available
                if rl_min_series and rl_max_series:
                    lines.append("```")
                    for (s1, mean_v), (_, min_v), (_, max_v) in zip(
                        rl_series, rl_min_series, rl_max_series
                    ):
                        lines.append(f"  step {s1:>3d}: mean={mean_v:.0f} range=[{min_v:.0f}, {max_v:.0f}]")
                    lines.append("```")
            else:
                lines.append("**Response Length**: No data.")
            lines.append("")

            # Clip ratio
            clip_series = sub.metric_series("response_length/clip_ratio")
            if clip_series:
                clip_vals = [v for _, v in clip_series]
                clip_avg = sum(clip_vals) / len(clip_vals)
                clip_max = max(clip_vals)
                clip_trend = sub.metric_trend("response_length/clip_ratio")
                warning = ""
                if clip_avg > 0.3:
                    warning = " [WARNING: high clip ratio, responses hitting token limit]"
                elif clip_avg > 0.15:
                    warning = " [moderate clipping]"
                lines.append(
                    f"**Clip Ratio**: avg={clip_avg:.3f}, max={clip_max:.3f}, "
                    f"trend={fmt_trend_arrow(clip_trend)}{warning}"
                )
                lines.append("```")
                for step, val in clip_series:
                    bar_len = int(val * 40)
                    lines.append(f"  step {step:>3d}: {val:.3f} {'#' * bar_len}")
                lines.append("```")
            lines.append("")

            # Entropy
            ent_series = sub.metric_series("actor/entropy_loss")
            if ent_series:
                ent_vals = [v for _, v in ent_series]
                ent_avg = sum(ent_vals) / len(ent_vals)
                ent_trend = sub.metric_trend("actor/entropy_loss")
                warning = ""
                if ent_avg < 0.1:
                    warning = " [WARNING: very low entropy, possible mode collapse]"
                elif ent_avg > 2.0:
                    warning = " [high entropy, diverse but possibly unfocused]"
                lines.append(
                    f"**Entropy**: avg={ent_avg:.3f}, trend={fmt_trend_arrow(ent_trend)}{warning}"
                )
            lines.append("")

            # Grad norm
            gn_series = sub.metric_series("actor/grad_norm")
            if gn_series:
                gn_vals = [v for _, v in gn_series]
                gn_avg = sum(gn_vals) / len(gn_vals)
                gn_max = max(gn_vals)
                gn_trend = sub.metric_trend("actor/grad_norm")
                warning = ""
                if gn_max > 10:
                    warning = " [WARNING: grad norm spike, training may be unstable]"
                lines.append(
                    f"**Grad Norm**: avg={gn_avg:.3f}, max={gn_max:.3f}, "
                    f"trend={fmt_trend_arrow(gn_trend)}{warning}"
                )
            lines.append("")

            # Memory
            mem_series = sub.metric_series("perf/max_memory_allocated_gb")
            if mem_series:
                mem_vals = [v for _, v in mem_series]
                mem_peak = max(mem_vals)
                mem_trend = sub.metric_trend("perf/max_memory_allocated_gb")
                lines.append(
                    f"**Memory**: peak={mem_peak:.1f} GB, "
                    f"range=[{min(mem_vals):.1f}, {mem_peak:.1f}] GB, "
                    f"trend={fmt_trend_arrow(mem_trend)}"
                )
            lines.append("")

            # Time per step
            time_series = sub.metric_series("timing_s/step")
            if time_series:
                time_vals = [v for _, v in time_series]
                time_avg = sum(time_vals) / len(time_vals)
                time_total = sum(time_vals)
                lines.append(
                    f"**Time/Step**: avg={time_avg:.1f}s, "
                    f"total training time={time_total / 3600:.1f}h"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_cross_experiment_insights(experiments: list) -> str:
    """Generate cross-experiment comparison and insights."""
    lines = []
    lines.append("## 3. Cross-Experiment Insights")
    lines.append("")

    # Collect all successful sub-experiments with their metrics
    all_subs = []
    for ef in experiments:
        for sub in ef.sub_experiments:
            if sub.status == "ok" and sub.final_score is not None:
                all_subs.append((ef, sub))

    if not all_subs:
        lines.append("No successful experiments found to analyze.")
        lines.append("")
        return "\n".join(lines)

    # --- Healthy training analysis ---
    lines.append("### Healthy Training Indicators")
    lines.append("")
    healthy = []
    anomalies = []

    for ef, sub in all_subs:
        issues = []

        # Check score improvement
        if sub.delta_score is not None and sub.delta_score < 0:
            issues.append("score regression")

        # Check response length stability
        rl_trend = sub.metric_trend("response_length/mean")
        if rl_trend == "up":
            rl_vals = [v for _, v in sub.metric_series("response_length/mean")]
            if rl_vals and max(rl_vals) > 800:
                issues.append(f"response length explosion (max={max(rl_vals):.0f})")

        # Check entropy
        ent_vals = [v for _, v in sub.metric_series("actor/entropy_loss")]
        if ent_vals:
            avg_ent = sum(ent_vals) / len(ent_vals)
            if avg_ent < 0.1:
                issues.append(f"entropy collapse ({avg_ent:.3f})")

        # Check grad norm
        gn_vals = [v for _, v in sub.metric_series("actor/grad_norm")]
        if gn_vals and max(gn_vals) > 10:
            issues.append(f"grad norm spike (max={max(gn_vals):.1f})")

        # Check clip ratio
        clip_vals = [v for _, v in sub.metric_series("response_length/clip_ratio")]
        if clip_vals and max(clip_vals) > 0.5:
            issues.append(f"high clip ratio (max={max(clip_vals):.2f})")

        if issues:
            anomalies.append((ef, sub, issues))
        else:
            healthy.append((ef, sub))

    lines.append(
        f"**{len(healthy)} sub-experiments** show healthy training "
        f"(score up, response length stable, entropy moderate, no grad spikes)."
    )
    if healthy:
        lines.append("")
        for ef, sub in sorted(healthy, key=lambda x: x[1].final_score or 0, reverse=True):
            model = sub.model.replace("Qwen/Qwen2.5-", "Q2.5-")
            lines.append(
                f"- **{sub.name}** ({model}, {sub.algorithm}): "
                f"score {fmt_pct(sub.init_score)} -> {fmt_pct(sub.final_score)} "
                f"({fmt_pct_change(sub.delta_score)})"
            )
    lines.append("")

    # --- Anomalies ---
    if anomalies:
        lines.append("### Anomalies Detected")
        lines.append("")
        for ef, sub, issues in anomalies:
            model = sub.model.replace("Qwen/Qwen2.5-", "Q2.5-")
            issues_str = ", ".join(issues)
            lines.append(
                f"- **{sub.name}** ({model}): {issues_str}"
            )
        lines.append("")

    # --- Best performing configurations ---
    lines.append("### Best Performing Configurations")
    lines.append("")

    # Sort by final score descending
    ranked = sorted(all_subs, key=lambda x: x[1].final_score or 0, reverse=True)

    lines.append("#### Top 10 by Final GSM8K Score")
    lines.append("")
    lines.append("| Rank | Sub | Model | Algo | Final Score | Delta | Steps |")
    lines.append("|------|-----|-------|------|-------------|-------|-------|")
    for rank, (ef, sub) in enumerate(ranked[:10], 1):
        model = sub.model.replace("Qwen/Qwen2.5-", "Q2.5-")
        lines.append(
            f"| {rank} | {sub.name} | {model} | {sub.algorithm} | "
            f"{fmt_pct(sub.final_score)} | {fmt_pct_change(sub.delta_score)} | "
            f"{sub.total_steps} |"
        )
    lines.append("")

    # --- Analysis by experiment category ---
    lines.append("### Group Comparisons")
    lines.append("")

    # Group by experiment ID and compute averages
    exp_groups = defaultdict(list)
    for ef, sub in all_subs:
        exp_groups[ef.exp_id].append((ef, sub))

    for exp_id in sorted(exp_groups.keys()):
        subs = exp_groups[exp_id]
        if len(subs) <= 1:
            continue

        ef_ref = subs[0][0]
        lines.append(f"**Exp {exp_id} ({ef_ref.exp_name.replace('-', ' ')})**:")
        lines.append("")

        scores = [sub.final_score for _, sub in subs if sub.final_score is not None]
        if scores:
            best_sub = max(subs, key=lambda x: x[1].final_score or 0)[1]
            worst_sub = min(subs, key=lambda x: x[1].final_score or 0)[1]
            avg_score = sum(scores) / len(scores)
            spread = max(scores) - min(scores)

            lines.append(f"- Average final score: {avg_score * 100:.1f}%")
            lines.append(
                f"- Best: {best_sub.name} ({best_sub.final_score * 100:.1f}%), "
                f"Worst: {worst_sub.name} ({worst_sub.final_score * 100:.1f}%)"
            )
            lines.append(f"- Spread: {spread * 100:.1f}pp")

            # Key differentiator
            if spread > 0.05:
                lines.append(
                    f"- Notable difference: {spread * 100:.1f}pp spread "
                    f"suggests the variable being tested has meaningful impact."
                )
            elif spread < 0.01:
                lines.append(
                    f"- Minimal spread: variable being tested has little impact "
                    f"within these conditions."
                )
        lines.append("")

    # --- Model scale analysis ---
    model_groups = defaultdict(list)
    for ef, sub in all_subs:
        model_groups[sub.model].append((ef, sub))

    if len(model_groups) > 1:
        lines.append("### Model Scale Comparison")
        lines.append("")
        lines.append("| Model | # Runs | Avg Score | Best Score |")
        lines.append("|-------|--------|-----------|------------|")
        for model in sorted(model_groups.keys()):
            runs = model_groups[model]
            scores = [s.final_score for _, s in runs if s.final_score is not None]
            if scores:
                model_name = model.replace("Qwen/Qwen2.5-", "Q2.5-")
                avg = sum(scores) / len(scores)
                best = max(scores)
                lines.append(f"| {model_name} | {len(scores)} | {avg * 100:.1f}% | {best * 100:.1f}% |")
        lines.append("")

    # --- Failed experiments summary ---
    failed_subs = []
    for ef in experiments:
        for sub in ef.sub_experiments:
            if sub.status != "ok":
                failed_subs.append((ef, sub))

    if failed_subs:
        lines.append("### Failed Experiments Summary")
        lines.append("")
        lines.append("| Exp | Sub | Status | Detail |")
        lines.append("|-----|-----|--------|---------|")
        for ef, sub in failed_subs:
            lines.append(
                f"| {ef.exp_id} | {sub.name} | {sub.status.upper()} | "
                f"{sub.error_detail[:60]} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_report(experiments: list) -> str:
    """Generate the full Markdown report."""
    parts = []

    parts.append("# TinyZero Experiment Analysis Report")
    parts.append("")
    parts.append(f"**Log directory**: `{DEFAULT_LOG_DIR}`")
    parts.append(f"**Files analyzed**: {len(experiments)}")

    total_subs = sum(len(ef.sub_experiments) for ef in experiments)
    ok_subs = sum(
        1 for ef in experiments for s in ef.sub_experiments if s.status == "ok"
    )
    failed_subs = total_subs - ok_subs
    parts.append(
        f"**Sub-experiments**: {total_subs} total "
        f"({ok_subs} OK, {failed_subs} failed/empty)"
    )
    parts.append("")

    # Summary stats
    all_scores = []
    for ef in experiments:
        for sub in ef.sub_experiments:
            if sub.final_score is not None:
                all_scores.append(sub.final_score)

    if all_scores:
        parts.append(f"**Score range**: {min(all_scores) * 100:.1f}% - {max(all_scores) * 100:.1f}%")
        parts.append(f"**Mean score**: {sum(all_scores) / len(all_scores) * 100:.1f}%")
    parts.append("")

    parts.append("---")
    parts.append("")

    parts.append(generate_summary_table(experiments))
    parts.append(generate_detailed_analysis(experiments))
    parts.append(generate_cross_experiment_insights(experiments))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR

    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found: {log_dir}", file=sys.stderr)
        print(f"Usage: {sys.argv[0]} [LOG_DIR]", file=sys.stderr)
        sys.exit(1)

    # Find all .out files
    out_files = sorted(
        str(p) for p in Path(log_dir).glob("exptz-*.out")
    )

    if not out_files:
        print(f"No log files found matching exptz-*.out in: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse all log files
    experiments = []
    for filepath in out_files:
        ef = parse_log_file(filepath)
        experiments.append(ef)

    # Sort by experiment ID then filename
    experiments.sort(key=lambda e: (e.exp_id.zfill(2), e.filename))

    # Generate and print report
    report = generate_report(experiments)
    print(report)


if __name__ == "__main__":
    main()
