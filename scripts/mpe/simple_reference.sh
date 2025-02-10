#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1
#SBATCH -p optimal
#SBATCH -A optimal
#SBATCH -J simple_reference
#SBATCH -o logs/simple_reference_0804-entropy0.001.out

## prepare for running
module load anaconda/2022.10
module load cuda/11.8
module load cudnn/8.7.0_cu11x

source activate mamba_nips24
export PYTHONUNBUFFERED=1

echo "current env"
which python

echo "Go to code dir"
cd /ailab/user/zhangyang2/Projects/Trans-mamba-nips24
pwd

map_name="simple_reference_v2"
env="pettingzoo"
seed=1
steps=1000000

# --ce_for_av
python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps $steps --mode offline --tokenizer vq --decay 0.8 \
                --temperature 2.0 --sample_temp inf --ce_for_av --ce_for_r