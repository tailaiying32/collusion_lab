#!/usr/bin/env bash
# Submit any CollusionLab single-run or sweep config as a SLURM job.
#
# Examples:
#   bash scripts/submit_slurm.sh configs/stego_capability_audit.yaml
#   bash scripts/submit_slurm.sh configs/sweep_stego_study.yaml --time 24:00:00 --cpus 8 --max-workers 8
#   bash scripts/submit_slurm.sh configs/sweep_stego_study.yaml --module miniforge --partition normal
#
# Secrets are not written into the job file. Keep OPENAI_API_KEY and
# COLLUSIONLAB_STORAGE_URL in your repo .env or exported in the submission shell.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/submit_slurm.sh CONFIG [options]

Options:
  --kind single|sweep|auto       Config type. Default: auto.
  --repo-root PATH              Repo root on cluster. Default: current repo root.
  --conda-env NAME              Conda/mamba env name. Default: collusion_lab.
  --module NAME                 Module to load before activation. Repeatable.
  --job-name NAME               SLURM job name. Default derives from config.
  --time HH:MM:SS               SLURM time limit. Default: 12:00:00.
  --cpus N                      SLURM cpus-per-task. Default: 4.
  --mem SIZE                    SLURM memory. Default: 16G.
  --partition NAME              SLURM partition.
  --account NAME                SLURM account.
  --qos NAME                    SLURM QoS.
  --constraint VALUE            SLURM constraint.
  --gres VALUE                  SLURM generic resources, e.g. gpu:1.
  --mail-user EMAIL             SLURM notification email.
  --mail-type VALUE             SLURM mail type. Default: END,FAIL.
  --dependency VALUE            SLURM dependency string.
  --max-workers N               Sweep workers. Default: --cpus.
  --output-dir PATH             Override CollusionLab output_dir.
  --run-id ID                   Single-run run_id override.
  --log-level LEVEL             Python log level. Default: INFO.
  --skip-storage-preflight      For sweeps, skip DB connectivity preflight.
  --job-dir PATH                Generated job file directory. Default: slurm_jobs.
  --log-dir PATH                SLURM stdout/stderr directory. Default: slurm_logs.
  --no-submit                   Write the sbatch file but do not submit it.
  -h, --help                    Show this help.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

shell_quote() {
    printf "%q" "$1"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_root="$(cd "$script_dir/.." && pwd)"

config=""
kind="auto"
repo_root="$default_repo_root"
conda_env="collusion_lab"
modules=()
job_name=""
time_limit="12:00:00"
cpus="4"
mem="16G"
partition=""
account=""
qos=""
constraint=""
gres=""
mail_user=""
mail_type="END,FAIL"
dependency=""
max_workers=""
output_dir=""
run_id=""
log_level="INFO"
skip_storage_preflight="false"
job_dir="slurm_jobs"
log_dir="slurm_logs"
no_submit="false"

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

config="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) kind="${2:?}"; shift 2 ;;
        --repo-root) repo_root="${2:?}"; shift 2 ;;
        --conda-env) conda_env="${2:?}"; shift 2 ;;
        --module) modules+=("${2:?}"); shift 2 ;;
        --job-name) job_name="${2:?}"; shift 2 ;;
        --time) time_limit="${2:?}"; shift 2 ;;
        --cpus|--cpus-per-task) cpus="${2:?}"; shift 2 ;;
        --mem) mem="${2:?}"; shift 2 ;;
        --partition) partition="${2:?}"; shift 2 ;;
        --account) account="${2:?}"; shift 2 ;;
        --qos) qos="${2:?}"; shift 2 ;;
        --constraint) constraint="${2:?}"; shift 2 ;;
        --gres) gres="${2:?}"; shift 2 ;;
        --mail-user) mail_user="${2:?}"; shift 2 ;;
        --mail-type) mail_type="${2:?}"; shift 2 ;;
        --dependency) dependency="${2:?}"; shift 2 ;;
        --max-workers) max_workers="${2:?}"; shift 2 ;;
        --output-dir) output_dir="${2:?}"; shift 2 ;;
        --run-id) run_id="${2:?}"; shift 2 ;;
        --log-level) log_level="${2:?}"; shift 2 ;;
        --skip-storage-preflight) skip_storage_preflight="true"; shift ;;
        --job-dir) job_dir="${2:?}"; shift 2 ;;
        --log-dir) log_dir="${2:?}"; shift 2 ;;
        --no-submit) no_submit="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$kind" == "auto" || "$kind" == "single" || "$kind" == "sweep" ]] || die "--kind must be auto, single, or sweep"

repo_root="$(cd "$repo_root" && pwd)"
if [[ "$config" = /* ]]; then
    config_path="$config"
else
    config_path="$repo_root/$config"
fi
[[ -f "$config_path" ]] || die "config not found: $config_path"

config_rel="$config_path"
case "$config_path" in
    "$repo_root"/*) config_rel="${config_path#"$repo_root"/}" ;;
esac

if [[ "$kind" == "auto" ]]; then
    if grep -Eq '^[[:space:]]*base_config:' "$config_path" && grep -Eq '^[[:space:]]*overrides:' "$config_path"; then
        kind="sweep"
    elif grep -Eq '^[[:space:]]*environment:' "$config_path" && grep -Eq '^[[:space:]]*agents:' "$config_path"; then
        kind="single"
    else
        die "could not infer config type from $config_path; pass --kind single or --kind sweep"
    fi
fi

if [[ -z "$job_name" ]]; then
    stem="$(basename "$config_path")"
    stem="${stem%.*}"
    stem="$(echo "$stem" | tr -c 'A-Za-z0-9_-' '_')"
    job_name="cl_${kind}_${stem}"
fi

if [[ -z "$max_workers" ]]; then
    max_workers="$cpus"
fi

if [[ "$job_dir" != /* ]]; then
    job_dir="$repo_root/$job_dir"
fi
if [[ "$log_dir" != /* ]]; then
    log_dir="$repo_root/$log_dir"
fi
mkdir -p "$job_dir" "$log_dir"

log_dir_arg="$log_dir"
case "$log_dir" in
    "$repo_root"/*) log_dir_arg="${log_dir#"$repo_root"/}" ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
job_path="$job_dir/${job_name}_${timestamp}.sbatch"

runner_cmd=(python -m)
if [[ "$kind" == "single" ]]; then
    runner_cmd+=(collusionlab.runner.experiment --config "$config_rel" --log-level "$log_level")
    [[ -n "$run_id" ]] && runner_cmd+=(--run-id "$run_id")
else
    runner_cmd+=(collusionlab.runner.sweep --sweep "$config_rel" --max-workers "$max_workers" --log-level "$log_level")
    [[ "$skip_storage_preflight" == "true" ]] && runner_cmd+=(--skip-storage-preflight)
fi
[[ -n "$output_dir" ]] && runner_cmd+=(--output-dir "$output_dir")

runner_cmd_quoted=""
for part in "${runner_cmd[@]}"; do
    runner_cmd_quoted+=" $(shell_quote "$part")"
done
runner_cmd_quoted="${runner_cmd_quoted# }"

{
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=$job_name"
    echo "#SBATCH --output=$log_dir_arg/%x_%j.out"
    echo "#SBATCH --error=$log_dir_arg/%x_%j.err"
    echo "#SBATCH --time=$time_limit"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=$cpus"
    echo "#SBATCH --mem=$mem"
    [[ -n "$partition" ]] && echo "#SBATCH --partition=$partition"
    [[ -n "$account" ]] && echo "#SBATCH --account=$account"
    [[ -n "$qos" ]] && echo "#SBATCH --qos=$qos"
    [[ -n "$constraint" ]] && echo "#SBATCH --constraint=$constraint"
    [[ -n "$gres" ]] && echo "#SBATCH --gres=$gres"
    [[ -n "$mail_user" ]] && echo "#SBATCH --mail-user=$mail_user"
    [[ -n "$mail_type" ]] && echo "#SBATCH --mail-type=$mail_type"
    [[ -n "$dependency" ]] && echo "#SBATCH --dependency=$dependency"
    echo "#SBATCH --export=ALL"
    echo
    echo "set -euo pipefail"
    echo
    echo 'echo "Job ${SLURM_JOB_ID:-unknown} started on $(hostname) at $(date)"'
    echo "echo \"Repo: $(shell_quote "$repo_root")\""
    echo "echo \"Config: $(shell_quote "$config_rel")\""
    echo "echo \"Kind: $kind\""
    echo
    for module_name in "${modules[@]}"; do
        echo "module load $(shell_quote "$module_name")"
    done
    echo
    echo "cd $(shell_quote "$repo_root")"
    echo
    cat <<EOF
if command -v mamba >/dev/null 2>&1; then
    eval "\$(mamba shell hook --shell bash)"
    mamba activate $(shell_quote "$conda_env")
elif command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    conda activate $(shell_quote "$conda_env")
elif [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate $(shell_quote "$conda_env")
elif [ -f "\$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "\$HOME/mambaforge/etc/profile.d/conda.sh"
    conda activate $(shell_quote "$conda_env")
else
    echo "Could not find mamba or conda. Try adding --module <cluster-module> or edit this job file." >&2
    exit 2
fi

export PYTHONPATH="src:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [ -n "\${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="\$SLURM_TMPDIR"
fi

echo "Python: \$(command -v python)"
echo "Starting command: $runner_cmd_quoted"
$runner_cmd_quoted
echo "Job \${SLURM_JOB_ID:-unknown} finished at \$(date)"
EOF
} > "$job_path"

chmod +x "$job_path"
echo "Wrote $job_path"

if [[ "$no_submit" == "true" ]]; then
    echo "Dry run only; not submitting to SLURM."
    exit 0
fi

sbatch "$job_path"
