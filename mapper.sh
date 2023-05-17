#!/bin/bash

source /etc/profile
module load anaconda/2023a
source activate chemprop

echo "Model: " $2
echo "Descriptor Bias: " $3
echo "Polynomial Bias: " $4
echo "Constant Bias: " $5
echo "Gauss Noise: " $6
echo "Split Type: " $7
echo "LF-HF Size ratio: " $8
echo "LF Superset of HF: " $9
echo "Seed: " ${10}


python multifidelity_end2end.py \
--model_type $2 \
--data_file /home/gridsan/kgreenman/mf_benchmark/chemprop/tests/data/gdb11_0.001.csv \
--hf_col_name h298 \
--lf_col_name h298_lf \
--scale_data True \
--save_test_plot True \
--num_epochs 30 \
--export_train_and_val False \
--add_descriptor_bias_to_make_lf $3 \
--add_pn_bias_to_make_lf $4 \
--add_constant_bias_to_make_lf $5 \
--add_gauss_noise_to_make_lf $6 \
--split_type $7 \
--lf_hf_size_ratio $8 \
--lf_superset_of_hf $9 \
--seed ${10} > ${11} 2>&1
