export https_proxy=http://10.7.4.2:3128
cd /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali
/root/miniconda3/envs/solo-learn/bin/python mobilenetv2_l3.py \
--num_gpus 8 \
--weight_decay 4e-5 \
--project Multi-Scale-CNN-HPO \
--num_workers 8 \
--batch_size 32 \
--checkpoint_dir checkpoints/mobilenetv2-l3-wd \
--run_name mobilenetv2-l3-wd \
--max_epochs 300 \
--learning_rate 0.045 \
--trunc 0.001
