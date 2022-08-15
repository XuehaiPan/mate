#!/usr/bin/env bash

#SBATCH --job-name      MATE-PSRO-NE
#SBATCH --nodes         1
#SBATCH --ntasks        1
#SBATCH --cpus-per-task 100
#SBATCH --gres          gpu:1
#SBATCH --qos           gpu
#SBATCH --partition     gpu
#SBATCH --time          3-00:00:00
#SBATCH --comment       "Multi-Agent Reinforcement Learning"
#SBATCH --output        slurm-%j-%x.stdout.txt
#SBATCH --error         slurm-%j-%x.stderr.txt

# === Print information ============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
	echo "Error: Bash is required to run this script." >&2
	exit 1
fi

function seperator() {
	echo
	printf '>%-.0s' $(seq 120)
	echo
	echo
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	for var in JOB_ID JOB_NAME NTASKS CPUS_ON_NODE CPUS_PER_TASK JOB_NODELIST JOB_PARTITION JOB_QOS; do
		var="SLURM_${var}"
		echo "${var}=${!var}" >&2
	done
	unset var
	seperator >&2
fi

# === Setup environment ============================================================================

export CONDA_HOME="${HOME}/Miniconda3" # change to your conda prefix
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_HOME}/lib"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate mate

if [[ -n "${WANDB_API_KEY:-}" && -x "$(command -v wandb)" ]]; then
	wandb login --relogin "${WANDB_API_KEY}"
fi

# === Jobs =========================================================================================

python3 -m examples.psro.train \
	--project mate-psro \
	--meta-solver NE \
	--num-workers 32 --num-envs-per-worker 8 --num-gpus 0.5 \
	--timesteps-total 5E6 --num-evaluation-episodes 100 --seed 0
