#!/usr/bin/env bash
# =============================================================================
# TinyZero — Submit round-2 fixed experiments
#
# Fixes: exp08_n16 (separate + 4h), exp10 LoRA (8h), exp13 1shot, exp14 distill, exp18 capability
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_EXP="$SCRIPT_DIR/experiment_fix.slurm"
LOG_DIR="/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs"
mkdir -p "$LOG_DIR"

# Time limits per experiment (hours)
declare -A TIME_LIMITS=(
    [08]=5 [10]=8 [13]=4 [14]=4 [18]=3
)

declare -A EXP_NAMES=(
    [08]="Rollout-N"
    [10]="LoRA-vs-Full"
    [13]="1shot-RLVR"
    [14]="Distill-vs-RL"
    [18]="Capability-Preservation"
)

EXPERIMENTS=(08 10 13 14 18)

echo "============================================================"
echo " TinyZero Round-2 Fix Pipeline"
echo " Experiments: ${EXPERIMENTS[*]}"
echo "============================================================"

PREV_JOB=""
SUBMITTED=()

for EXP_ID in "${EXPERIMENTS[@]}"; do
    TIME_H=${TIME_LIMITS[$EXP_ID]:-4}
    NAME=${EXP_NAMES[$EXP_ID]:-"Exp-$EXP_ID"}

    DEP_ARG=""
    if [ -n "$PREV_JOB" ]; then
        DEP_ARG="--dependency=afterany:$PREV_JOB"
    fi

    JOB=$(sbatch --parsable \
        $DEP_ARG \
        --export=ALL,EXP_ID=$EXP_ID \
        --job-name="tz-${EXP_ID}-${NAME}" \
        --time="${TIME_H}:00:00" \
        "$SLURM_EXP")

    echo "  Exp $EXP_ID ($NAME): job $JOB  [${TIME_H}h]${DEP_ARG:+ $DEP_ARG}"
    PREV_JOB=$JOB
    SUBMITTED+=("$EXP_ID:$JOB")
done

echo ""
echo "============================================================"
echo " All ${#SUBMITTED[@]} jobs submitted!"
for ENTRY in "${SUBMITTED[@]}"; do
    IFS=':' read -r EID JID <<< "$ENTRY"
    echo "   Exp $EID (${EXP_NAMES[$EID]}): $JID"
done
echo ""
echo " Monitor: squeue -u \$USER"
echo "============================================================"

MANIFEST="$LOG_DIR/job_manifest_fix2_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "# TinyZero Round-2 Fix Job Manifest — $(date)"
    for ENTRY in "${SUBMITTED[@]}"; do
        IFS=':' read -r EID JID <<< "$ENTRY"
        echo "exp${EID}=$JID  # ${EXP_NAMES[$EID]}"
    done
} > "$MANIFEST"
echo "Manifest saved: $MANIFEST"
