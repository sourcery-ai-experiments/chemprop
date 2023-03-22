#!/bin/bash

source /etc/profile
module load anaconda/2023a
source activate chemprop

echo "Model: " $2
echo "Descripter Bias: " $3
echo "Polynomial Bias: " $4
echo "Gauss Noise: " $5
echo "Size ratio: " $6
echo "LF superset of HF: " $7


python multifidelity_end2end.py --model_type $2 --add_descriptor_bias_to_make_lf $3 --add_pn_bias_to_make_lf $4 --add_gauss_noise_to_make_lf $5 --lf_hf_size_ratio $6 --lf_superset_of_hf $7