export https_proxy=http://10.7.4.2:3128
/root/miniconda3/envs/solo-learn/bin/python resnet50.py \
  --num_gpus 2 \
  --weight_decay 2e-5 \
  --project msc-transfer-linear \
  --num_workers 8 \
  --batch_size 32 \
  --dataset_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/ \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/resnet50-baseline/last.ckpt \
  --run_name resnet50 \
  --max_epochs 90 \
  --learning_rate 0.1
