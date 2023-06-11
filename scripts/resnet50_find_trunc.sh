export https_proxy=http://10.7.4.2:3128
cd /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali
/root/miniconda3/envs/solo-learn/bin/python resnet50_l2_norm.py \
  --num_gpus 8 \
  --weight_decay 2e-5 \
  --project Multi-Scale-CNN-HPO \
  --num_workers 8 \
  --batch_size 128 \
  --checkpoint_dir checkpoints/resnet50-l2-norm \
  --run_name resnet50-l2-norm \
  --max_epochs 90 \
  --learning_rate 0.5
