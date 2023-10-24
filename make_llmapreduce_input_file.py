command_str_list = [
    "000 single_fidelity 0 0 0 0 random 1 True 0",
    "001 single_fidelity 0 0 0 0 random 1 True 1",
    "002 single_fidelity 0 0 0 0 random 1 True 2",
]
i = 3

for seed in [0, 1, 2]:
    for model_type in [
        "multi_target",
        "multi_fidelity",
        "multi_fidelity_weight_sharing",
    ]:
        for split_type in [
            "random",
            # "scaffold",
            # "h298",
            # "molwt",
            # "atom",
        ]:
            for lf_hf_size_ratio in [1, 10, 100]:
                for lf_superset_of_hf in [
                    True,
                    False,
                ]:
                    for descriptor_bias in [0, 10, 50]:
                        command_str = f"{i:03} {model_type} {descriptor_bias} 0 0 0 {split_type} {lf_hf_size_ratio} {lf_superset_of_hf} {seed}"
                        command_str_list.append(command_str)
                        i += 1
                    for polynomial_bias in [0, 1, 2, 3]:
                        command_str = f"{i:03} {model_type} 0 {polynomial_bias} 0 0 {split_type} {lf_hf_size_ratio} {lf_superset_of_hf} {seed}"
                        command_str_list.append(command_str)
                        i += 1
                    for constant_bias in [0, 10, 50]:
                        command_str = f"{i:03} {model_type} 0 0 {constant_bias} 0 {split_type} {lf_hf_size_ratio} {lf_superset_of_hf} {seed}"
                        command_str_list.append(command_str)
                        i += 1
                    for gauss_noise in [0, 10, 50]:
                        command_str = f"{i:03} {model_type} 0 0 0 {gauss_noise} {split_type} {lf_hf_size_ratio} {lf_superset_of_hf} {seed}"
                        command_str_list.append(command_str)
                        i += 1

command_str_list = list(set(command_str_list))

with open("inputs_orthogonal.txt", "w") as f:
    for i, command_str in enumerate(command_str_list):
        if i == 0:
            f.write(command_str)
        else:
            f.write("\n" + command_str)