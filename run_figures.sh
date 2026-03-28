#!/bin/bash
#SBATCH --partition=dev
#SBATCH --time=00:05:00
#SBATCH --job-name=diffpir-figs
#SBATCH --output=figures_%j.out

source /home/azhuang/DiffPIR/.venv/bin/activate
cd /home/azhuang/DiffPIR
python make_figures.py --output-dir outputs_ffhq --figures-dir figures_ffhq
