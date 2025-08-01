import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Dataset(Dataset):
    def __init__(self, csv_path, root_dir, split="train", img_size=256, augment=False):
        """
        Args:
            csv_path (str): CSV file path
            root_dir (str): Root directory where image & mask files are stored
            split (str): 'train' or 'test'
            img_size (int): Target size for resizing
            augment (bool): Whether to apply augmentation (only recommended for train)
        """
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.transform = (
            A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            if self.augment and split == "train"
            else A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.root_dir, row["image_path"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Binary label: 1 = defect, 0 = normal
        label = 0 if row["is_normal"] == 1 else 1

        # Load mask if exists
        if pd.isna(row["gt_mask"]) or row["gt_mask"] == "none":
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask_path = os.path.join(self.root_dir, row["gt_mask"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = (mask > 0).astype(np.float32)

        # Apply transforms (image + mask)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0)  # shape: [1, H, W]

        return image, label, mask


def get_dataloaders(csv_path, root_dir, batch_size=8, img_size=256):
    train_dataset = Dataset(
        csv_path, root_dir, split="train", img_size=img_size, augment=True
    )
    test_dataset = Dataset(
        csv_path, root_dir, split="test", img_size=img_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


if __name__ == "__main__":
    csv_file = "path/to/dataset.csv"  # 替换为你的 CSV 路径
    root_dir = "path/to/dataset"  # 替换为图像和掩膜文件所在的根目录

    train_loader, test_loader = get_dataloaders(
        csv_file, root_dir, batch_size=4, img_size=256
    )

    print("Train Loader Length:", len(train_loader))
    print("Test Loader Length:", len(test_loader))

    # 测试一个 batch
    for images, labels, masks in train_loader:
        print("Image batch shape:", images.shape)  # [B, 3, H, W]
        print("Label batch shape:", labels.shape)  # [B]
        print("Mask batch shape:", masks.shape)  # [B, 1, H, W]
        break
