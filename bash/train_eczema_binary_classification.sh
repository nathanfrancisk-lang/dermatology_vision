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
--batch_size 32 \
--n_height 224 \
--n_width 224 \
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
--learning_rates 1e-4 \
--learning_rate_scheduler cosine_annealing \
--warmup_updates 500 \
--warmup_init_lr 1e-7 \
--adam_betas 0.9 0.999 \
--weight_decay 1e-4 \
--n_epoch 45 \
--loss_func cross_entropy \
--label_smoothing 0.05 \
--classification_threshold 0.5 \
--target_sensitivity 0.9 \
--checkpoint_dirpath \
    checkpoints/eczema_binary_resnet50_32x224x224_ce_ls0.05_sens0.9_valsplit \
--n_step_per_checkpoint 4000 \
--n_step_per_summary 100 \
--n_display 5 \
--start_step_validation 0 \
--early_stop_patience 0 \
--device gpu \
--n_thread 4

# Validation is validation/dermnet (2002 images), not testing/dermnet. Checkpoint selection and
# the calibrated threshold both come off this set, so it must stay disjoint from whatever gets
# reported. setup/split_val_test.py builds it; testing/dermnet/dermnet_holdout_* is the other half.
#
# Plain --loss_func cross_entropy, no --loss_class_weights. --balance_sampler already yields 50/50
# batches; the old 1.0/3.0 weight on top only compressed the probability scale (calibrated
# thresholds fell to ~0.07) without improving ranking, which is all specificity_at_sensitivity
# reads. Recall pressure belongs in the threshold, applied after training, for free.
#
# --early_stop_patience 0 (disabled) and --n_epoch 45: patience 10 previously fired at step
# 36000/48400, mid-cosine, while val loss was still falling — it was counting noise in the
# selection metric. 45 epochs (49815 steps at 35403 training images) lets cosine anneal to
# completion with room past where tensorboard showed the old run still improving. model-best.pth
# already keeps the best checkpoint, so running to the end costs time and nothing else.
#
# --n_step_per_checkpoint 4000 gives 12 validations instead of 36. Fewer draws from a noisy
# metric means less inflation of the selected maximum, and 12 x 282MB of checkpoints instead of 36.
#
# Training is 35403 images, not the earlier 38718: 20% of fitzpatrick17k is now held out as a
# skin-tone stratified test set (testing/fitzpatrick17k), which the setup script builds.
#
# --learning_schedule is not passed: cosine_annealing ignores it entirely (T_max is derived from
# n_epoch). It only applies to --learning_rate_scheduler multi_step.
