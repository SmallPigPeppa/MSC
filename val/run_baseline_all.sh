export https_proxy=http://10.7.4.2:3128
export PYTHONPATH=/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/resnet50-baseline/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name resnet50-baseline\
#  --method resnet50

/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all.py \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/vgg16-bn-baseline/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val\
  --run_name vgg16-bn-baseline\
  --method vgg16_bn


/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all.py \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/densenet121-baseline/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val\
  --run_name densenet121-baseline\
  --method densenet121

/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all.py \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/baseline/checkpoints/mobilenetv2-baseline/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val\
  --run_name mobilenetv2-baseline\
  --method mobilenetv2

