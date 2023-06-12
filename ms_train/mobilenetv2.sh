export https_proxy=http://10.7.4.2:3128
cd /mnt/mmtech01/usr/liuwenzhuo/code/test-code/MSC-dali/ms_train
/root/miniconda3/envs/solo-learn/bin/python mobilenetv2.py \
--num_gpus 8 \
--weight_decay 2e-5 \
--project Multi-Scale-CNN-HPO \
--num_workers 8 \
--batch_size 32 \
--checkpoint_dir checkpoints/mobilenetv2-mstrain \
--run_name mobilenetv2-mstrain \
--max_epochs 300 \
--learning_rate 0.1 \
--trunc 0.001