#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_eczema_binary_classification.py \
--train_image_paths_file \
    training/dermnet/dermnet_train_image.txt \
    training/fitzpatrick17k/fitzpatrick17k_train_image.txt \
    training/sd198/sd198_train_image.txt \
--train_labels_file \
    training/dermnet/dermnet_train_binary_label.txt \
    training/fitzpatrick17k/fitzpatrick17k_train_binary_label.txt \
    training/sd198/sd198_train_binary_label.txt \
--val_image_paths_file \
    testing/dermnet/dermnet_test_image.txt \
--val_labels_file \
    testing/dermnet/dermnet_test_binary_label.txt \
--batch_size 32 \
--n_height 224 \
--n_width 224 \
--normalized_image_range 0 1 \
--random_brightness 0.8 1.2 \
--random_contrast 0.8 1.2 \
--random_saturation 0.8 1.2 \
--random_hue -0.1 0.1 \
--random_flip_type horizontal \
--random_rotate_max 15 \
--random_crop \
--balance_sampler \
--encoder_type resnet50 \
--pretrained \
--learning_rates 1e-4 \
--learning_schedule 50 80 \
--learning_rate_scheduler cosine_annealing \
--warmup_updates 500 \
--warmup_init_lr 1e-7 \
--adam_betas 0.9 0.999 \
--weight_decay 1e-4 \
--n_epoch 100 \
--loss_func cross_entropy \
--classification_threshold 0.35 \
--checkpoint_dirpath \
    checkpoints/eczema_binary_resnet50_32x224x224_lr1e-4_crossentropy_thresh0.35 \
--n_step_per_checkpoint 1000 \
--n_step_per_summary 100 \
--n_display 5 \
--start_step_validation 0 \
--device gpu \
--n_thread 2
