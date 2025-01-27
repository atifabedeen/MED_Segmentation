#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a40|a100|m40
#SBATCH --mem-per-gpu=20G

module load conda/latest
source /work/pi_pkatz_umass_edu/atif_experiments/segmentation/hss/bin/activate

cd /work/pi_pkatz_umass_edu/atif_experiments/segmentation

python -m scripts.training_UNETR
