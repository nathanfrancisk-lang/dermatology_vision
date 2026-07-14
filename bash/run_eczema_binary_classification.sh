#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_eczema_binary_classification.py \
--image_paths_file \
    testing/dermnet/dermnet_test_image.txt \
--labels_file \
    testing/dermnet/dermnet_test_binary_label.txt \
--checkpoint_path \
    checkpoints/eczema_binary_resnet50/model-best.pth \
--batch_size 32 \
--n_height 224 \
--n_width 224 \
--normalized_image_range 0 1 \
--encoder_type resnet50 \
--output_path \
    results/eczema_binary_resnet50 \
--save_outputs \
--device gpu \
--n_thread 8
