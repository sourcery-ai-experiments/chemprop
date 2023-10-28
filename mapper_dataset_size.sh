#!/bin/bash

source /etc/profile
module load anaconda/2023a
source activate chemprop

echo "Model: " $2
echo "Dataset Size: " $3
echo "Descriptor Bias: " $4
echo "Polynomial Bias: " $5
echo "Constant Bias: " $6
echo "Gauss Noise: " $7
echo "Split Type: " $8
echo "LF-HF Size ratio: " $9
echo "LF Superset of HF: " ${10}
echo "Seed: " ${11}


python multifidelity_end2end.py \
--model_type $2 \
--data_file /home/gridsan/kgreenman/mf_benchmark/chemprop/tests/data/gdb11_$3.csv \
--hf_col_name h298 \
--lf_col_name h298_lf \
--scale_data True \
--save_test_plot True \
--num_epochs 30 \
--export_train_and_val False \
--add_descriptor_bias_to_make_lf $4 \
--add_pn_bias_to_make_lf $5 \
--add_constant_bias_to_make_lf $6 \
--add_gauss_noise_to_make_lf $7 \
--split_type $8 \
--lf_hf_size_ratio $9 \
--lf_superset_of_hf ${10} \
--seed ${11} > ${12} 2>&1
