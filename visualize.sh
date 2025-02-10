marie_path="xx.pth"
tokenizer="vq"

mamba_path="xx.pth"

# 2m_vs_1z
map_name="2s3z"
env="starcraft"

CUDA_VISIBLE_DEVICES=1 python visualize.py --env ${env} --map_name ${map_name} --tokenizer ${tokenizer} \
                    --model_path ${marie_path} \
                    --eval_episodes 10 \
                    --ce_for_r --ce_for_av --temperature 0.5