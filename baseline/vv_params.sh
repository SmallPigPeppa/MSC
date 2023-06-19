export https_proxy=http://10.7.4.2:3128
cd /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline
/root/miniconda3/envs/solo-learn/bin/python vv_params.py \
  --num_gpus 8 \
  --weight_decay 2e-5 \
  --project vv_params \
  --num_workers 8 \
  --batch_size 128 \
  --checkpoint_dir vv_params \
  --run_name resnet50-baseline \
  --max_epochs 1 \
  --learning_rate 0.5
