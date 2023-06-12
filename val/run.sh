export https_proxy=http://10.7.4.2:3128
/root/miniconda3/envs/solo-learn/bin/python  msc_val_flops.py \
  --checkpoint_path checkpoints/resnet50-l2-norm/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val\
  --run_name resnet50-l2\
