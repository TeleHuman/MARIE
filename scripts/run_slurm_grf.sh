set -x
PARTITION=${PARTITION:-"optimal"}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

seeds=(
    2 3 4
)
# academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper
env="football"
map_name="academy_pass_and_shoot_with_keeper"
steps=2000000

for seed in "${seeds[@]}"; do
    date_dir=$(date "+%Y-%m-%d")
    cur_date=$(date "+%H-%M-%S")
    OUTPUT_DIR=training-runs/$date_dir

    # log_name=$(echo "$map_name" | awk -F'_' '{print $(NF-2) "-" $(NF-1) "-" $NF}')
    log_name="$map_name-seed_$seed"
    echo $log_name
    note=

    mkdir -p $OUTPUT_DIR

    sbatch -p ${PARTITION} \
    -J ${log_name}${note:+-$note} \
    -N 1 \
    -n 6 \
    -o ${OUTPUT_DIR}/${cur_date}-${log_name}${note:+-$note}-%j.out \
    --gres=gpu:${GPUS_PER_NODE} \
    --wrap="python train.py \
            --n_workers 1 \
            --env $env \
            --env_name $map_name \
            --seed $seed \
            --steps $steps \
            --mode offline \
            --tokenizer fsq \
            --decay 0.8 \
            --temperature 2.0 \
            --sample_temp inf \
            --ce_for_r"
done