#!/bin/bash

source /etc/profile
module load anaconda/2023a
source activate chemprop-mf

echo "Model: " $2
echo "Loss Modifier" $3
echo "Seed: " $4


python multifidelity_end2end.py \
--model_type $2 \
--data_file /home/gridsan/torkhon/chemprop-mf/tests/data/gdb11_0.01.csv \
--hf_col_name h298 \
--lf_col_name h298_lf \
--scale_data True \
--save_test_plot True \
--num_epochs 30 \
--export_train_and_val False \
--loss_mod $3\
--seed $4\
--add_gauss_noise_to_make_lf $5 2>&1
