"""Run the dual-branch ResNet-50 on an external test cohort.

Loads the weights produced by `finetune_naic.py`, runs inference over the
cohort listed in the config and writes per-sample predictions to the output
directory given in the config.

Usage:
    python inference.py --config configs/inference.yaml
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import yaml


from models.dual_resnet50_model import DualResnet50Classifier
from train import make_dataset, make_loader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(weights_path, device):
    """Load the fine-tuned weights into the model and switch to eval mode."""
    model = DualResnet50Classifier(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(model, loader, device):
    """Return sample ids, true labels and positive-class scores for a loader."""
    ctids, y_true, y_score = [], [], []
    with torch.no_grad():
        for primary_img, lymph_img, label, ctid in loader:
            primary_img = primary_img.to(device, dtype=torch.float)
            lymph_img = lymph_img.to(device, dtype=torch.float)
            score = torch.softmax(model(primary_img, lymph_img), dim=1)[:, 1]

            ctids.extend(ctid)
            y_true.extend(label.cpu().numpy())
            y_score.extend(score.cpu().numpy())
    return ctids, np.asarray(y_true), np.asarray(y_score)

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run the dual-branch ResNet-50 on a cohort."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"model weights not found: {config['model_path']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    test_df = pd.read_excel(config["data_file"], sheet_name=config["sheet_name"])
    test_df = test_df.rename(columns={"CT_id": "ctid", "Label": "label"})
    test_df = test_df[["ctid", "label"]]
    logger.info(
        "loaded %d samples | class counts %s",
        len(test_df), dict(test_df["label"].value_counts().sort_index()),
    )

    test_ds = make_dataset(test_df, config["output_dir"], "test", config, test=True)
    test_loader = make_loader(test_ds, config, shuffle=False)

    model = load_model(config["model_path"], device)
    ctids, y_true, y_score = predict(model, test_loader, device)


    pd.DataFrame(
        {
            "ctid": ctids,
            "true_label": y_true,
            "pred_prob": y_score,
        }
    ).to_csv(os.path.join(config["output_dir"], "predictions.csv"), index=False)
    logger.info("done. results in %s", config["output_dir"])


if __name__ == "__main__":
    main()
