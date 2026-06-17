#!/bin/bash

echo " Starting SD-198 Dataset Processing"

python setup/setup_dataset_sd198.py \
--root_dir \
    data/sd198 \
--output_dir \
    data/sd198_derived \