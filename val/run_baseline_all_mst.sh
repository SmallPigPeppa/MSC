export https_proxy=http://10.7.4.2:3128
export PYTHONPATH=/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali
/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all_mst.py \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/mstrain/checkpoints/resnet50-mstrain/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val\
  --run_name resnet50-mst\
  --method resnet50
#
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all_mst.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/mstrain/checkpoints/densenet121-mstrain/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name densenet121-mst\
#  --method densenet121
#
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all_mst.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/mstrain/checkpoints/vgg16-bn-mstrain/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name vgg16-bn-mst\
#  --method vgg16_bn
#
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_baseline_all_mst.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/mstrain/checkpoints/mobilenetv2-mstrain/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name mobilenetv2-mst\
#  --method mobilenetv2
