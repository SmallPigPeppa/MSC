export https_proxy=http://10.7.4.2:3128
cd /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline
/root/miniconda3/envs/solo-learn/bin/python resnet50.py \
  --num_gpus 8 \
  --weight_decay 2e-5 \
  --project Multi-Scale-CNN-HPO \
  --num_workers 8 \
  --batch_size 128 \
  --checkpoint_dir checkpoints/resnet50-baseline \
  --run_name resnet50-baseline \
  --max_epochs 90 \
  --learning_rate 0.5
