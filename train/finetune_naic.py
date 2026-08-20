"""Fine-tune the NAC-pretrained dual-branch ResNet-50 on the NAIC cohort.

Starts from the weights produced by `train.py` with `configs/pretrain_nac.yaml`
and fine-tunes them with layer-wise learning rates. Stratified k-fold
cross-validation is run for evaluation, followed by a final fit on the full
dataset. Per-fold splits, epoch metrics, per-epoch predictions and the final
weights are written to the output directory given in the config.

Usage:
    python finetune_naic.py --config configs/finetune_naic.yaml
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold

from models.dual_resnet50_model import DualResnet50Classifier
from train import (
    collect_preds,
    evaluate,
    make_dataset,
    make_loader,
    preds_to_frame,
    setup_seed,
    train_one_epoch,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Parameter groups for layer-wise learning rates. Keys are matched as
# substrings of the parameter name and the first matching group wins.
EARLY_LAYER_KEYS = ("conv1", "bn1", "layer1", "layer2")
LATE_LAYER_KEYS = ("layer3", "layer4")


# --------------------------------------------------------------------------- #
# Pretrained weights
# --------------------------------------------------------------------------- #
def load_pretrained_weights(model, weights_path, device):
    """Copy every tensor whose name and shape match the target model."""
    pretrained = torch.load(weights_path, map_location=device)
    target = model.state_dict()
    matched = {
        k: v
        for k, v in pretrained.items()
        if k in target and v.shape == target[k].shape
    }
    if not matched:
        raise ValueError(
            f"no tensor in {weights_path} matches the model; check that the "
            "checkpoint was produced by the same architecture"
        )
    target.update(matched)
    model.load_state_dict(target)
    logger.info(
        "loaded %d/%d pretrained tensors from %s",
        len(matched), len(pretrained), weights_path,
    )
    return model


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_model(config, device):
    model = DualResnet50Classifier(
        pretrained=True, num_classes=2, dropout_rate=config["dropout_rate"]
    )
    load_pretrained_weights(model, config["pretrained_weights"], device)
    return model.to(device)


def layer_groups(model):
    early, late, head = [], [], []
    for name, param in model.named_parameters():
        lowered = name.lower()
        if any(key in lowered for key in EARLY_LAYER_KEYS):
            early.append(param)
        elif any(key in lowered for key in LATE_LAYER_KEYS):
            late.append(param)
        else:
            head.append(param)
    return early, late, head


def build_optimizer(model, config):
    early, late, head = layer_groups(model)
    multipliers = config["lr_multipliers"]
    optimizer = torch.optim.AdamW(
        [
            {"params": early, "lr": config["lr"] * multipliers["early"]},
            {"params": late, "lr": config["lr"] * multipliers["late"]},
            {"params": head, "lr": config["lr"] * multipliers["head"]},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )
    return optimizer, scheduler


# --------------------------------------------------------------------------- #
# Core training loop (shared by CV folds and the full-dataset run)
# --------------------------------------------------------------------------- #
def run_training(train_loader, val_loader, config, out_dir, device, n_epochs):
    """Train for n_epochs, writing metrics/predictions/weights to out_dir.

    If val_loader is given, the model is evaluated every epoch and the
    final-epoch validation AUC is returned; otherwise NaN is returned.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = build_optimizer(model, config)

    metrics, train_store, val_store = [], {}, {}
    final_val_auc = float("nan")

    for epoch in range(n_epochs):
        train_loss, train_auc, tr_ids, tr_true, tr_score = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        collect_preds(train_store, tr_ids, tr_true, tr_score, epoch)

        lr_early, lr_late, lr_head = scheduler.get_last_lr()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "lr_early": lr_early,
            "lr_late": lr_late,
            "lr_head": lr_head,
        }

        if val_loader is not None:
            val_auc, va_ids, va_true, va_score = evaluate(
                model, val_loader, device, epoch
            )
            collect_preds(val_store, va_ids, va_true, va_score, epoch)
            row["val_auc"] = val_auc
            final_val_auc = val_auc
            logger.info(
                "epoch %d/%d | loss %.4f | train_auc %.4f | val_auc %.4f",
                epoch + 1, n_epochs, train_loss, train_auc, val_auc,
            )
        else:
            logger.info(
                "epoch %d/%d | loss %.4f | train_auc %.4f",
                epoch + 1, n_epochs, train_loss, train_auc,
            )

        metrics.append(row)
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(out_dir, f"epoch{n_epochs}.pth"))
    pd.DataFrame(metrics).to_csv(
        os.path.join(out_dir, "training_metrics.csv"), index=False
    )
    preds_to_frame(train_store, n_epochs).to_csv(
        os.path.join(out_dir, "train_pred.csv"), index=False
    )
    if val_loader is not None:
        preds_to_frame(val_store, n_epochs).to_csv(
            os.path.join(out_dir, "val_pred.csv"), index=False
        )
    return final_val_auc


# --------------------------------------------------------------------------- #
# Cross-validation and full-dataset training
# --------------------------------------------------------------------------- #
def cross_validate(full_df, config, device):
    skf = StratifiedKFold(
        n_splits=config["n_folds"], shuffle=True, random_state=config["seed"]
    )
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        setup_seed(config["seed"] + fold)
        fold_dir = os.path.join(config["output_dir"], f"fold_{fold}")

        train_ds = make_dataset(
            full_df.iloc[train_idx].reset_index(drop=True),
            fold_dir, f"train_fold_{fold}", config, test=False,
        )
        val_ds = make_dataset(
            full_df.iloc[val_idx].reset_index(drop=True),
            fold_dir, f"val_fold_{fold}", config, test=True,
        )
        train_loader = make_loader(
            train_ds, config, shuffle=True, seed=config["seed"] + fold
        )
        val_loader = make_loader(val_ds, config, shuffle=False)

        logger.info(
            "fold %d/%d | train %d | val %d",
            fold + 1, config["n_folds"], len(train_ds), len(val_ds),
        )
        fold_aucs.append(
            run_training(train_loader, val_loader, config, fold_dir, device,
                         config["epochs"])
        )
        torch.cuda.empty_cache()

    return fold_aucs


def train_full(full_df, config, device):
    setup_seed(config["seed"])
    full_dir = os.path.join(config["output_dir"], "full_dataset")
    full_ds = make_dataset(full_df, full_dir, "full_dataset", config, test=False)
    full_loader = make_loader(full_ds, config, shuffle=True, seed=config["seed"])
    logger.info("full dataset | n = %d", len(full_ds))
    run_training(full_loader, None, config, full_dir, device,
                 config["full_dataset_epochs"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the NAC-pretrained dual-branch ResNet-50 "
                    "on the NAIC cohort."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["pretrained_weights"]):
        raise FileNotFoundError(
            f"pretrained weights not found: {config['pretrained_weights']}"
        )

    setup_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    full_df = pd.read_excel(config["data_file"], sheet_name=config["sheet_name"])
    full_df = full_df.rename(columns={"CT_id": "ctid", "Label": "label"})
    full_df = full_df[["ctid", "label"]]
    logger.info(
        "loaded %d samples | class counts %s",
        len(full_df), dict(full_df["label"].value_counts().sort_index()),
    )

    fold_aucs = cross_validate(full_df, config, device)
    mean_auc, std_auc = float(np.nanmean(fold_aucs)), float(np.nanstd(fold_aucs))

    summary = pd.DataFrame(
        [{"fold": i + 1, "val_auc": a} for i, a in enumerate(fold_aucs)]
        + [{"fold": "mean", "val_auc": mean_auc},
           {"fold": "std", "val_auc": std_auc}]
    )
    summary.to_csv(os.path.join(config["output_dir"], "cv_summary.csv"), index=False)
    logger.info("cross-validation AUC: %.4f +/- %.4f", mean_auc, std_auc)

    train_full(full_df, config, device)
    logger.info("done. results in %s", config["output_dir"])


if __name__ == "__main__":
    main()