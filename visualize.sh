marie_path="/mnt/data/optimal/zhangyang/Projects/reprod_w_old/0507_repro_results/starcraft/2s3z_vq/run1/ckpt/model_final.pth"
# marie_path="/mnt/data/optimal/zhangyang/Projects/reprod_w_old/0503_repro_results/starcraft/3s_vs_3z_vq/run1/ckpt/model_final.pth"
tokenizer="vq"

mamba_path="/mnt/data/optimal/zhangyang/code/mamba/mamba_results/starcraft/2s_vs_1sc/run1/ckpt/mamba_model.pth"

# 2m_vs_1z
map_name="2s3z"
env="starcraft"

CUDA_VISIBLE_DEVICES=1 python visualize.py --env ${env} --map_name ${map_name} --tokenizer ${tokenizer} \
                    --model_path ${marie_path} \
                    --eval_episodes 10 \
                    --ce_for_r --ce_for_av --temperature 0.5