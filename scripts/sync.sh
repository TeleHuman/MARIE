#!/bin/bash

# 指定要查找的前缀字符串
PREFIX="wandb/offline-run-202409"

# 遍历所有以指定前缀开头的文件和目录
for dir in ${PREFIX}*; do
    # 检查是否为目录
    if [ -d "$dir" ]; then
        # 输出目录路径
        # echo "$dir"
        wandb sync $dir
    fi
done
