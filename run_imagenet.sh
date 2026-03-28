#!/bin/bash
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=diffpir-imagenet
#SBATCH --output=imagenet_%j.out

source /home/azhuang/DiffPIR/.venv/bin/activate
cd /home/azhuang/DiffPIR

export HF_HOME=/home/azhuang/.cache/huggingface
export HF_HUB_CACHE=/home/azhuang/.cache/huggingface/hub
export HF_DATASETS_CACHE=/home/azhuang/.cache/huggingface/datasets

python experiments.py \
    --device cuda \
    --output-dir outputs_imagenet \
    --image-dir testsets/imagenet_val_100 \
    --tasks gaussian_blur motion_blur inpainting_box inpainting_random \
    --methods diffpir_hqs_diffunet diffpir_drs_diffunet dpir dps
