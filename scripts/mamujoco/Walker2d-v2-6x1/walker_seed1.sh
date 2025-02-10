#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1
#SBATCH -p optimal
#SBATCH -A optimal
#SBATCH -J Walker2d-v2-6x1_0822
#SBATCH -o Walker2d-v2-6x1_0823.out

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
# pwd

## map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42) so_many_baneling MMM 2s3z
agent_conf="6x1"
scenario="Walker2d-v2"
env="mamujoco"
seed=1
steps=2000000

# --ce_for_av
python train.py --n_workers 1 --env ${env} --env_name ${scenario} --seed ${seed} --agent_conf ${agent_conf} \
                --steps $steps --mode offline --tokenizer vq --decay 0.8 \
                --temperature 1.0 --sample_temp inf --ce_for_end