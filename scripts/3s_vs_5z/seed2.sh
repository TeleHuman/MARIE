#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1
#SBATCH -p optimal
#SBATCH -A optimal
#SBATCH -J 1M_seed2_3s5z

## prepare for running
module load anaconda/2022.10
module load cuda/11.8
module load cudnn/8.6.0_cu11x

source activate mamba
export PYTHONUNBUFFERED=1

echo "current env"
which python

echo "Go to code dir"
cd /ailab/user/zhangyang2/Projects/Trans-mamba
pwd

## map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42) so_many_baneling MMM 2s3z
map_name="3s_vs_5z"
env="starcraft"
seed=2
steps=200000

# --ce_for_av
python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps $steps --mode offline --tokenizer vq --decay 0.8 \
                --temperature 2.0 --sample_temp inf --ce_for_av 