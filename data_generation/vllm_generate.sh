#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=02:00:00
#SBATCH --job-name=vllm_llama
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --output=vllm_llama_%j.out
#SBATCH --error=vllm_llama_%j.err
#SBATCH --mail-user=<YOUR_NORTHEASTERN_EMAIL>
#SBATCH --mail-type=ALL

#modules
module load anaconda3/2024.06
module load cuda/12.8.0

# change to workspace dir
cd ~/workspace

#python env
source .venv/bin/activate

unset http_proxy https_proxy

echo "running python script"
#code
python generate_vllm.py

# capture exit code
EXIT_CODE=$?

echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
