#!/usr/bin/env bash
#SBATCH --job-name=cl_sweep_sweep_stego_study_
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-unknown} started on $(hostname) at $(date)"
echo "Repo: /home/tty6/collusion_lab"
echo "Config: configs/sweep_stego_study.yaml"
echo "Kind: sweep"


cd /home/tty6/collusion_lab

if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate collusion_lab
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate collusion_lab
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate collusion_lab
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
    conda activate collusion_lab
else
    echo "Could not find mamba or conda. Try adding --module <cluster-module> or edit this job file." >&2
    exit 2
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="$SLURM_TMPDIR"
fi

echo "Python: $(command -v python)"
echo "Starting command: python -m collusionlab.runner.sweep --sweep configs/sweep_stego_study.yaml --max-workers 4 --log-level INFO"
python -m collusionlab.runner.sweep --sweep configs/sweep_stego_study.yaml --max-workers 4 --log-level INFO
echo "Job ${SLURM_JOB_ID:-unknown} finished at $(date)"
