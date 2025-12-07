#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=dolphin_engagement_farmer_dataset
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=dolphin_engagement_farmer_dataset_%j.out
#SBATCH --error=dolphin_engagement_farmer_dataset_%j.err
#SBATCH --mail-user=magadum.a@northeastern.edu
#SBATCH --mail-type=ALL

#modules
#module load python/3.13
module load anaconda3/2024.06
module load cuda/12.8.0

#python env
source /home/magadum.a/workspace/.venv/bin/activate

cd /scratch/$USER
#mkdir ollama_models_scratch

#apptainer pull ollama.sif docker://ollama/ollama
unset http_proxy https_proxy

echo "running ollama"
export APPTAINERENV_OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch
export OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch
export OLLAMA_NUM_PARALLEL=4

apptainer run --nv -B "/projects:/projects,/scratch:/scratch" ollama.sif serve &

sleep 10

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1


echo "running python script"
#code
python /home/magadum.a/workspace/generate.py

# capture exit code
EXIT_CODE=$?

# stop apptainer instances -a for all instances of $USER (this doesn't work on HPC)
# apptainer instance stop -a


echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
