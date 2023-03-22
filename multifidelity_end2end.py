import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chemprop.v2 import data, featurizers, models
from chemprop.v2.models import modules


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.model_type == "single_fidelity" and args.add_bias_to_make_lf > 0:
        raise ValueError("Cannot add bias to make low fidelity data when model type is single fidelity")

    if args.model_type == "multi_fidelity_weight_sharing_non_diff":
        raise NotImplementedError("Not implemented yet")

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # Model
    mgf = featurizers.MoleculeFeaturizer()
    mp_block_hf = modules.molecule_block()  # TODO: use aggregation='sum' or 'norm' instead of default 'mean'?
    mp_block_lf = modules.molecule_block()  # TODO: use aggregation='sum' or 'norm' instead of default 'mean'?

    model_dict = {
        "single_fidelity": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "multi_target": models.RegressionMPNN(mp_block_hf, n_tasks=2),  # TODO: multi-target regression
        "multi_fidelity": models.MultifidelityRegressionMPNN(mp_block_hf, n_tasks=1, mpn_block_low_fidelity=mp_block_lf),  # TODO: multi-fidelity without weight sharing
        "multi_fidelity_weight_sharing": models.MultifidelityRegressionMPNN(mp_block_hf, n_tasks=1),  # multi-fidelity with weight sharing
        # "multi_fidelity_weight_sharing_non_diff": ,  # TODO: multi-fidelity non-differentiable feature
    }
    # TODO: add option for multi-fidelity with evidential uncertainty?

    mpnn = model_dict[args.model_type]

    # Data
    data_df = pd.read_csv(args.data_file)
    smis = data_df["smiles"].tolist()

    if args.add_bias_to_make_lf > 0 or args.add_noise_to_make_lf > 0:
        noise = np.random.normal(args.add_bias_to_make_lf, args.add_noise_to_make_lf, len(data_df))
        data_df[args.lf_col_name] = data_df[args.hf_col_name] + noise

    if args.model_type == "single_fidelity":
        targets = data_df[[args.hf_col_name]].values.reshape(-1, 1)
    else:
        targets = data_df[[args.lf_col_name, args.hf_col_name]].values

    all_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(smis, targets)]

    train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=0)
    train_data, val_data = train_test_split(train_data, test_size=0.11, random_state=0)

    train_dset = data.MoleculeDataset(train_data, mgf)
    val_dset = data.MoleculeDataset(val_data, mgf)
    test_dset = data.MoleculeDataset(test_data, mgf)

    if args.scale_data:
        train_scaler = train_dset.normalize_targets()
        _ = val_dset.normalize_targets(train_scaler)
        test_scaler = test_dset.normalize_targets()

    train_loader = data.MolGraphDataLoader(train_dset, batch_size=50, num_workers=12)
    val_loader = data.MolGraphDataLoader(val_dset, batch_size=50, num_workers=12, shuffle=False)
    test_loader = data.MolGraphDataLoader(test_dset, batch_size=50, num_workers=12, shuffle=False)

    # Train
    trainer = pl.Trainer(
        # logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1,
        max_epochs=args.num_epochs,
    )
    trainer.fit(mpnn, train_loader, val_loader)

    preds = trainer.predict(mpnn, test_loader)

    test_smis = [x.smi for x in test_data]

    if args.model_type == "single_fidelity":
        preds = [x[0].item() for x in preds]

        targets = [x.targets[0] for x in test_data]

        if args.scale_data:
            preds = test_scaler.inverse_transform(np.array(preds).reshape(-1, 1))
            targets = test_scaler.inverse_transform(np.array(targets).reshape(-1, 1))
        else:
            preds = np.array(preds)
            targets = np.array(targets)

        print("Test set")
        mae, rmse, r2 = eval_metrics(targets, preds)

        if args.save_test_plot:
            plt.scatter(targets, preds)
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            plt.text(min(targets), max(preds),
                     f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR^2: {r2:.2f}",
                     fontsize=12, ha='left', va='top')
            plt.savefig("mf_test_preds.png")

        test_df = pd.DataFrame(
            {
                "smiles": test_smis,
                args.hf_col_name: targets.flatten(),
                f"{args.hf_col_name}_preds": preds.flatten(),
            }
        )
    else:
        if args.model_type == "multi_target":
            preds = np.array([x[0].numpy()[0] for x in preds])
        elif args.model_type == "multi_fidelity" or args.model_type == "multi_fidelity_weight_sharing":
            preds = np.array([[[x[0][0].numpy(), x[0][1].numpy()]] for x in preds]).reshape(len(preds), 2)
        else:
            raise ValueError("Not implemented yet")  # TODO: multi-fidelity non-differentiable

        preds = test_scaler.inverse_transform(preds)

        preds_lf, preds_hf = preds[:, 0], preds[:, 1]  # TODO: are HF and LF ordered the same for multi-fidelity? this should be correct for multi-target

        # Both HF and LF targets are identical if the only difference in the original HF and LF was a bias term -- this is not a bug -- once normalized, the network should learn both the same way
        targets = [x.targets for x in test_data]
        targets = test_scaler.inverse_transform(targets)

        targets_lf, targets_hf = targets[:, 0], targets[:, 1]

        # TODO: unscale preds_{h,l}f and targets_{h,l}f for multi-fidelity

        print("High Fidelity - Test set")
        hf_mae, hf_rmse, hf_r2 = eval_metrics(targets_hf, preds_hf)
        print("Low Fidelity - Test set")
        lf_mae, lf_rmse, lf_r2 = eval_metrics(targets_lf, preds_lf)

        if args.save_test_plot:
            fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=2)

            axes[0].scatter(targets_hf, preds_hf, alpha=0.3, label="High Fidelity")
            axes[0].set_xlabel("Target")
            axes[0].set_ylabel("Prediction")
            axes[0].text(min(targets_hf), max(preds_hf),
                     f"MAE: {hf_mae:.2f}\nRMSE: {hf_rmse:.2f}\nR^2: {hf_r2:.2f}",
                     fontsize=12, ha='left', va='top')
            axes[0].set_title("High Fidelity")

            axes[1].scatter(targets_lf, preds_lf, alpha=0.3, label="Low Fidelity")
            axes[1].set_xlabel("Target")
            axes[1].set_ylabel("Prediction")
            axes[1].text(min(targets_lf), max(preds_lf),
                     f"MAE: {lf_mae:.2f}\nRMSE: {lf_rmse:.2f}\nR^2: {lf_r2:.2f}",
                     fontsize=12, ha='left', va='top')
            axes[1].set_title("Low Fidelity")

            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            plt.savefig("mf_test_preds.png", bbox_inches="tight")

        test_df = pd.DataFrame(
            {
                "smiles": test_smis,
                args.hf_col_name: targets_hf.flatten(),
                f"{args.hf_col_name}_preds": preds_hf.flatten(),
                args.lf_col_name: targets_lf.flatten(),
                f"{args.lf_col_name}_preds": preds_lf.flatten(),
            }
        )

    test_df.to_csv("mf_test_preds.csv", index=False)

    if args.export_train_and_val:
        train_smis = [x.smi for x in train_data]
        val_smis = [x.smi for x in val_data]
        train_targets = np.array([x.targets[0] for x in train_data])
        val_targets = np.array([x.targets[0] for x in val_data])

        if args.scale_data:
            train_targets = train_scaler.inverse_transform(train_targets.reshape(-1, 1))
            val_targets = train_scaler.inverse_transform(val_targets.reshape(-1, 1))

        train_df = pd.DataFrame(
            {
                "smiles": train_smis,
                args.hf_col_name: train_targets.flatten(),
            }
        )
        val_df = pd.DataFrame(
            {
                "smiles": val_smis,
                args.hf_col_name: val_targets.flatten(),
            }
        )
        train_df.to_csv("mf_train.csv", index=False)
        val_df.to_csv("mf_val.csv", index=False)


def eval_metrics(targets, preds):
    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)
    r2 = r2_score(targets, preds)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    return mae, rmse, r2


def add_args(parser: ArgumentParser):
    parser.add_argument("--model_type", type=str, default="single_fidelity", choices=["single_fidelity", "multi_target", "multi_fidelity", "multi_fidelity_weight_sharing", "multi_fidelity_non_diff"])
    parser.add_argument("--data_file", type=str, default="tests/data/gdb11_0.001.csv")  # choices=["multifidelity_joung_stda_tddft.csv", "gdb11_0.0001.csv" (too small), "gdb11_0.0001.csv"]
    parser.add_argument("--hf_col_name", type=str, default="h298")  # choices=["h298", "lambda_maxosc_tddft"]
    parser.add_argument("--lf_col_name", type=str, default="h298_bias_1", required=False)  # choices=["h298_bias_1", "lambda_maxosc_stda"]
    parser.add_argument("--scale_data", action="store_true")
    parser.add_argument("--save_test_plot", action="store_true")
    parser.add_argument("--add_bias_to_make_lf", type=float, default=0.0)
    parser.add_argument("--add_noise_to_make_lf", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--export_train_and_val", action="store_true")
    return


if __name__ == "__main__":
    main()
