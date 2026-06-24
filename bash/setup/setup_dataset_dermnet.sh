#!/bin/bash

echo " Starting Dermnet Dataset Processing"

python setup/setup_dataset_dermnet.py \
--root_dir \
    data/dermnet \
--output_dir \
    data/dermnet_derived \
