#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Point this at whichever training run won on VALIDATION AUC. Comparing runs on the held-out
# split would quietly turn it into a third validation set; it is read once, here, at the end.
RUN_NAME=eczema_binary_resnet50_16x288x288_lr7e-5_ce_ls0.05_sens0.9_valsplit
CHECKPOINT_DIR=checkpoints/${RUN_NAME}
RESULTS_DIR=results/${RUN_NAME}

N_HEIGHT=288
N_WIDTH=288

# Three evaluation sets, three different jobs:
#   validation/dermnet      calibrates thresholds, never reported
#   testing/dermnet holdout the headline numbers
#   testing/fitzpatrick17k  the skin tone breakdown, the only set carrying Fitzpatrick labels
#
# Crop geometry is not passed: run() reads it from the checkpoint, so inference cannot silently
# preprocess differently than training did. N_HEIGHT/N_WIDTH still have to match the run.

for SPLIT in "val validation/dermnet/dermnet_val" \
             "holdout testing/dermnet/dermnet_holdout" \
             "fitzpatrick testing/fitzpatrick17k/fitzpatrick17k_test"; do
    set -- ${SPLIT}
    NAME=$1
    PREFIX=$2

    python src/run_eczema_binary_classification.py \
    --image_paths_file \
        ${PREFIX}_image.txt \
    --labels_file \
        ${PREFIX}_binary_label.txt \
    --checkpoint_path \
        ${CHECKPOINT_DIR}/model-best.pth \
    --threshold_path \
        ${CHECKPOINT_DIR}/threshold.txt \
    --batch_size 32 \
    --n_height ${N_HEIGHT} \
    --n_width ${N_WIDTH} \
    --normalized_image_range 0 1 \
    --encoder_type resnet50 \
    --output_path \
        ${RESULTS_DIR}/${NAME} \
    --save_outputs \
    --device gpu \
    --n_thread 8
done

# Headline report, led by the rule-out operating point. Also covers discrimination, the
# malignancy safety budget, sensitivity targets, and false positives by true diagnosis.
# Every threshold is calibrated on val; every number is measured on holdout.
python validity/report_operating_points.py \
--val_probabilities_file \
    ${RESULTS_DIR}/val/probabilities.txt \
--val_labels_file \
    validation/dermnet/dermnet_val_binary_label.txt \
--val_original_labels_file \
    validation/dermnet/dermnet_val_original_label.txt \
--test_probabilities_file \
    ${RESULTS_DIR}/holdout/probabilities.txt \
--test_labels_file \
    testing/dermnet/dermnet_holdout_binary_label.txt \
--test_original_labels_file \
    testing/dermnet/dermnet_holdout_original_label.txt \
--target_npv 0.98 \
--max_malignant_leakage 0.05 \
--target_sensitivities 0.85 0.90 0.95 \
--n_bootstrap 2000 \
--output_path \
    ${RESULTS_DIR}

# Skin tone report. Same threshold, applied unchanged to a different dataset, so this doubles as
# a cross-dataset generalization check.
python validity/report_operating_points.py \
--val_probabilities_file \
    ${RESULTS_DIR}/val/probabilities.txt \
--val_labels_file \
    validation/dermnet/dermnet_val_binary_label.txt \
--test_probabilities_file \
    ${RESULTS_DIR}/fitzpatrick/probabilities.txt \
--test_labels_file \
    testing/fitzpatrick17k/fitzpatrick17k_test_binary_label.txt \
--test_scales_file \
    testing/fitzpatrick17k/fitzpatrick17k_test_fitzpatrick_scale.txt \
--target_npv 0.98 \
--target_sensitivities 0.85 0.90 0.95 \
--n_bootstrap 2000 \
--output_path \
    ${RESULTS_DIR}/fitzpatrick

# Grad-CAM overlays. Reuses the probabilities computed above rather than re-scoring, and covers
# false positives and false negatives, not only the cases that worked.
python validity/grad_cam.py \
--image_paths_file \
    testing/dermnet/dermnet_holdout_image.txt \
--labels_file \
    testing/dermnet/dermnet_holdout_binary_label.txt \
--probabilities_file \
    ${RESULTS_DIR}/holdout/probabilities.txt \
--checkpoint_path \
    ${CHECKPOINT_DIR}/model-best.pth \
--threshold_path \
    ${CHECKPOINT_DIR}/threshold.txt \
--n_height ${N_HEIGHT} \
--n_width ${N_WIDTH} \
--normalized_image_range 0 1 \
--encoder_type resnet50 \
--n_sample 6 \
--output_path \
    ${RESULTS_DIR}/grad_cam \
--device gpu
