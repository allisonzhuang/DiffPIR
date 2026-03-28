#!/bin/bash
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=diffpir-ffhq
#SBATCH --output=ffhq_%j.out

source /home/azhuang/DiffPIR/.venv/bin/activate
cd /home/azhuang/DiffPIR

python experiments.py \
    --device cuda \
    --output-dir outputs_ffhq \
    --image-dir testsets/ffhq_val_100 \
    --tasks gaussian_blur motion_blur inpainting_box inpainting_random \
    --methods diffpir_hqs_diffunet diffpir_drs_diffunet dpir dps
