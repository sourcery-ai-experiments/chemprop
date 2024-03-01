command_str_list = []

for seed in [0, 1, 2]:
    for model_type in ["multi_fidelity", "evidentialmf", "mvemultifidelity"]:
        for loss_mod in [2, 5, 10, 30, 50]:
            for gauss_noise in [5]:
                command_str = f"{model_type} {loss_mod} {seed} {gauss_noise}"
                command_str_list.append(command_str)


with open("inputs_unc.txt", "w") as f:
    for i, command_str in enumerate(command_str_list):
        if i == 0:
            f.write(command_str)
        else:
            f.write("\n" + command_str)
