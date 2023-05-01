#!/bin/bash
#SBATCH -J single_fidelity
#SBATCH -o single_fidelity-%j.out
#SBATCH -t 1-00:00:00
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --mem=30gb
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2023a
source activate chemprop

echo "Model: single_fidelity"
echo "Descriptor Bias: 0"
echo "Polynomial Bias: 0"
echo "Constant Bias: 0"
echo "Gauss Noise: 0"
echo "Split Type: random"
echo "LF-HF Size ratio: 1"
echo "LF Superset of HF: True"


python multifidelity_end2end.py \
--model_type single_fidelity \
--data_file /home/gridsan/kgreenman/mf_benchmark/chemprop/tests/data/gdb11_0.001.csv \
--hf_col_name h298 \
--lf_col_name h298_lf \
--scale_data True \
--save_test_plot True \
--num_epochs 30 \
--export_train_and_val False \
--add_descriptor_bias_to_make_lf 0 \
--add_pn_bias_to_make_lf 0 \
--add_constant_bias_to_make_lf 0 \
--add_gauss_noise_to_make_lf 0 \
--split_type random \
--lf_hf_size_ratio 1 \
--lf_superset_of_hf True \
--seed 0 > output/000.out 2>&1
