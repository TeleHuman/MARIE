marie_path="xx.pth"
tokenizer="vq"

mamba_path="xx.pth"

# 2m_vs_1z
map_name="2m_vs_1z"
env="starcraft"

CUDA_VISIBLE_DEVICES=0 python validate_model.py --env ${env} --env_name ${map_name} --tokenizer ${tokenizer} \
                                             --marie_load_path ${marie_path} \
                                             --mamba_load_path ${mamba_path}