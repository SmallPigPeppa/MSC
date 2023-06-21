#!/bin/bash

export https_proxy=http://10.7.4.2:3128
export PYTHONPATH=/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali

# 设置要运行的数据集
#datasets=("cifar10" "cifar100" "stl10" "caltech" "fashion" "flowers" "pets" "cars" "rafdb" "aircraft" "dtd")
datasets=("cifar10")

# 设置你要运行的模型及其对应的脚本和检查点路径
declare -A models
models["resnet50"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/resnet50-l2-last/last.ckpt"
#models["densenet121"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/densenet121-l2-seprate-trans/last.ckpt"
#models["vgg16"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/vgg16-l2-new/last.ckpt"
#models["mobilenetv2"]="/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/mobilenetv2-l3-wd2/last.ckpt"

# 设置每个模型对应的GPU卡号
declare -A gpus
gpus["resnet50"]="0,1"
#gpus["densenet121"]="2,3"
#gpus["vgg16"]="4,5"
#gpus["mobilenetv2"]="6,7"

# 迭代不同的数据集
for dataset in "${datasets[@]}"; do
  # 迭代不同的模型
  for model in "${!models[@]}"; do
    checkpoint=${models[$model]}
    script="main_msc.py"
    gpu=${gpus[$model]}
    # 运行脚本
    CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/solo-learn/bin/python $script \
      --num_gpus 2 \
      --weight_decay 2e-5 \
      --project msc-transfer-linear-msc \
      --num_workers 8 \
      --batch_size 32 \
      --dataset_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/ \
      --checkpoint_path $checkpoint \
      --run_name msc \
      --max_epochs 90 \
      --learning_rate 0.1 \
      --model $model \
      --dataset $dataset &
  done
  wait
done
