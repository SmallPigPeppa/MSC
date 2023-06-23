#    elif args.method == 'densenet121':
#        #checkpoints/densenet121-l2-seprate-trans
#        from densenet121_l2_seprate_trans import DenseNet121_L2
#        model = DenseNet121_L2.load_from_checkpoint(args.checkpoint_path, args=args)
#    elif args.method == 'vgg16':
#        #checkpoints/vgg16-l2-new
#        from vgg16_l3 import VGG16_L2
#        model = VGG16_L2.load_from_checkpoint(args.checkpoint_path, args=args)
#    elif args.method == 'mobilenetv2':
#        #checkpoints/mobilenetv2-l3-wd2
#        from mobilenetv2_l3 import MobileNetV2_L3
#        model = MobileNetV2_L3.load_from_checkpoint(args.checkpoint_path, args=args)
export https_proxy=http://10.7.4.2:3128
export PYTHONPATH=/mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali
/root/miniconda3/envs/solo-learn/bin/python  msc_val_flops.py \
  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/resnet50-l2-last/last.ckpt\
  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --project msc-val-debug\
  --run_name resnet50-l2\

#/root/miniconda3/envs/solo-learn/bin/python  msc_val_flops_all.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/densenet121-l2-seprate-trans/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name densenet121-l2\
#  --method densenet121
#
#
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_flops_all.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/vgg16-l2-new/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name vgg16-l2 \
#  --method vgg16
#
#/root/miniconda3/envs/solo-learn/bin/python  msc_val_flops_all.py \
#  --checkpoint_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/checkpoints/mobilenetv2-l3-wd2/last.ckpt\
#  --dataset_path /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --project msc-val\
#  --run_name mobilenetv2-l2\
#    --method mobilenetv2

