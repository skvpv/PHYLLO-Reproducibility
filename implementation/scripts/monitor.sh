#!/usr/bin/env bash
# Lightweight run monitor for CP-02 (single GPU). Non-destructive.
echo "== squeue (this user) =="; squeue --me 2>/dev/null || echo "slurm not available"
echo "== GPU =="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi n/a"
echo "== latest work dirs =="; find "${PHYLLO_WORK_ROOT:-.}" -name 'iter_*.pth' -printf '%T+ %p\n' 2>/dev/null | sort | tail -10
