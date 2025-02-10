#!/bin/bash
# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42) so_many_baneling MMM 2s3z
map_name="3s_vs_5z"
env="starcraft"
seed=2
cuda_device=0
steps=2000

# --ce_for_av
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps $steps --mode disabled --tokenizer vq --decay 0.8 \
                                                    --temperature 1.0 --sample_temp inf --ce_for_av