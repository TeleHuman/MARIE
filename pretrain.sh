# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42)
map_name="so_many_baneling"
env="starcraft"

CUDA_VISIBLE_DEVICES=1 python pretrain_wm.py --env ${env} --map ${map_name} --num_steps 30000 \
                                             --mode online \
                                             --tokenizer vq --decay 0.8 \
                                             --epochs 500 --ce_for_av --ce_for_end