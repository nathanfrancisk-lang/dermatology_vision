#!/bin/bash

echo " Starting Fitzpatrick17k Dataset Processing"

python setup/setup_dataset_fitzpatrick17k.py \
--root_dir \
    data/fitzpatrick17k \
--output_dir \
    data/fitzpatrick17k_derived \
