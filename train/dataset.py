import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StomachCancerDataset(Dataset):
    """Dual-branch (primary tumour + lymph node) CT dataset.

    Args:
        csv_file: Path to an .xlsx/.csv file with `ctid` and `label` columns.
        dataset_root: Directory containing one sub-folder per `ctid`.
        test: If True, use the deterministic evaluation transform.
    """

    def __init__(self, csv_file, dataset_root, test=False):
        try:
            self.df = pd.read_excel(csv_file)
        except Exception:
            self.df = pd.read_csv(csv_file, encoding="ANSI")

        self.names = [str(i) for i in self.df["ctid"]]
        self.dataset_root = dataset_root
        self.test = test

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        ctid = self.names[idx]
        label = self.df.iloc[idx]["label"]

        primary_path = os.path.join(
            self.dataset_root, ctid, "primary_tumor_processed.nii.gz"
        )
        lymph_path = os.path.join(
            self.dataset_root, ctid, "lymph_node_processed.nii.gz"
        )

        primary_img = self.load_nii_file(primary_path)
        lymph_img = self.load_nii_file(lymph_path)

        transform = self.test_transform if self.test else self.train_transform
        primary_img = transform(primary_img)
        lymph_img = transform(lymph_img)

        return primary_img, lymph_img, label, ctid

    def load_nii_file(self, file_path):
        """Load a 3-channel NIfTI volume as a uint8 `[H, W, C]` array."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        img_array = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        if img_array.ndim != 3:
            raise ValueError(f"expected a 3-D array, got shape {img_array.shape}")

        img_array = img_array.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()

        # ToPILImage requires uint8 [H, W, C].
        img_array = (img_array * 255).astype(np.uint8)
        img_array = np.transpose(img_array, (1, 2, 0))
        return img_array


def get_data_loaders(csv_file, dataset_root, batch_size=8, num_workers=4):
    """Build train/validation loaders sharing a single labelled spreadsheet."""
    train_dataset = StomachCancerDataset(
        csv_file=csv_file, dataset_root=dataset_root, test=False
    )
    val_dataset = StomachCancerDataset(
        csv_file=csv_file, dataset_root=dataset_root, test=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader