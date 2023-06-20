#!/bin/bash

export https_proxy=http://10.7.4.2:3128

# 设置要运行的数据集
datasets=("cifar100" "stl10")

# 设置你要运行的模型及其对应的脚本和检查点路径
declare -A models
models["resnet50"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/resnet50-baseline/last.ckpt"
models["densenet121"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/densenet121-baseline/last.ckpt"
models["vgg16"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/vgg16-bn-baseline/last.ckpt"
models["mobilenetv2"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/mobilenetv2-baseline/last.ckpt"

# 设置每个模型对应的GPU卡号
declare -A gpus
gpus["resnet50"]="0,1"
gpus["densenet121"]="2,3"
gpus["vgg16"]="4,5"
gpus["mobilenetv2"]="6,7"

# 迭代不同的数据集
for dataset in "${datasets[@]}"; do
    # 迭代不同的模型
    for model in "${!models[@]}"; do
        checkpoint=${models[$model]}
        script="main.py"
        gpu=${gpus[$model]}
        # 运行脚本
        CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/solo-learn/bin/python $script \
            --num_gpus 2 \
            --weight_decay 2e-5 \
            --project msc-transfer-linear \
            --num_workers 8 \
            --batch_size 32 \
            --dataset_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/ \
            --checkpoint_path $checkpoint \
            --run_name linear \
            --max_epochs 90 \
            --learning_rate 0.1 \
            --dataset $dataset &
    done
    wait
done
