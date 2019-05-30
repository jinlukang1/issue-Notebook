#!/bin/bash
cd /ghome/jinlk/jinlukang/example/examples/fast_neural_style/trainDVSC && \
python trainnet.py \
        --selection False \
        --dataroot /gdata/jinlk/jinlukang/example/DVSC/train/ \
        --train_batch_size 32 \
        --train_shuffle True \
        --train_num_workers 8 \
        --train_epoch 2 \
        --model_save_path /gdata/jinlk/jinlukang/example/DVSC/TrainedModel/ \
        --use_tensorboard False \
        --tblog_path /ghome/jinlk/jinlukang/example/examples/fast_neural_style/trainDVSC/tblog/ \
        --train_load_path /ghome/jinlk/jinlukang/example/examples/fast_neural_style/trainDVSC/checkpoints/vgg16-397923af.pth \
        --train_lr 0.001

        
