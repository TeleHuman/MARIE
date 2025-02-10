marie_path="xx.pth"

tokenizer="fsq"

# 2m_vs_1z
map_name="academy_3_vs_1_with_keeper"
env="football"

python visualize_replay.py --env ${env} --env_name ${map_name} --tokenizer ${tokenizer} \
                    --model_path ${marie_path} \
                    --eval_episodes 10 \
                    --ce_for_r --temperature 1.0