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
    validation/dermnet/dermnet_val_image.txt \
--val_labels_file \
    validation/dermnet/dermnet_val_binary_label.txt \
--batch_size 16 \
--n_height 288 \
--n_width 288 \
--normalized_image_range 0 1 \
--random_brightness 0.8 1.2 \
--random_contrast 0.8 1.2 \
--random_saturation 0.8 1.2 \
--random_hue -0.1 0.1 \
--random_flip_type horizontal vertical \
--random_rotate_max 30 \
--random_crop \
--balance_sampler \
--encoder_type resnet50 \
--pretrained \
--learning_rates 7e-5 \
--learning_rate_scheduler cosine_annealing \
--warmup_updates 500 \
--warmup_init_lr 1e-7 \
--adam_betas 0.9 0.999 \
--weight_decay 1e-4 \
--n_epoch 25 \
--loss_func cross_entropy \
--label_smoothing 0.05 \
--classification_threshold 0.5 \
--target_sensitivity 0.9 \
--checkpoint_dirpath \
    checkpoints/eczema_binary_resnet50_16x288x288_lr7e-5_ce_ls0.05_sens0.9_valsplit \
--n_step_per_checkpoint 4500 \
--n_step_per_summary 100 \
--n_display 5 \
--start_step_validation 0 \
--early_stop_patience 0 \
--device gpu \
--n_thread 4

# Same config as train_eczema_binary_classification.sh, at 288px. Source images have median
# 480-1130px on the long side, so 224 was discarding real lesion texture; resolution is the
# largest remaining lever on this hardware.
#
# --batch_size 16 is forced by the 4GB RTX 3050 Ti: 288^2/224^2 = 1.65x activation memory, so the
# batch has to halve. --learning_rates 7e-5 is partial linear scaling for the smaller batch.
# --n_epoch 25 at 2213 steps/epoch keeps this near 6h; --n_step_per_checkpoint 4500 keeps it at
# 12 validations, matching the 224px run. Fewer epochs than the 224px run purely for wall clock:
# each step costs ~1.65x more pixels.
#
# If this OOMs (it will inside the first ~20 steps or not at all), drop to --n_height 256
# --n_width 256 (1.31x) before cutting batch size further.
