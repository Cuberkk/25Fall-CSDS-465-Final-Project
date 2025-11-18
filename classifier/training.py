#!/usr/bin/env python
# coding: utf-8

import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.classification import BinaryF1Score


# =============================
# Hyper-parameters & paths
# =============================

BATCH = 16
LR = 1e-4
IM_SIZE = 299

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 根目录：
DATA_ROOT = "./plant-pathology-2021-fgvc8"
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DATA_FILE = os.path.join(DATA_ROOT, "train.csv")

BESTMODEL_DIR = os.path.join(DATA_ROOT, "inception_v3_bestmodel")


# =============================
# 读取 & 处理标签
# =============================

def read_image_labels(csv_path: str) -> pd.DataFrame:
    """Read train.csv and index by image name."""
    df = pd.read_csv(csv_path).set_index("image")
    return df


def get_single_labels(unique_labels) -> List[str]:
    """Split multi-label strings and return list of unique classes."""
    single_labels = []
    for label in unique_labels:
        single_labels += label.split()
    single_labels = set(single_labels)
    return list(single_labels)


def get_one_hot_encoded_labels(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'labels' column into multi-hot columns."""
    df = dataset_df.copy()
    unique_labels = df.labels.unique()
    column_names = get_single_labels(unique_labels)

    # 初始化列
    df[column_names] = 0

    # one-hot / multi-hot
    for label in unique_labels:
        label_indices = df[df["labels"] == label].index
        splited_labels = label.split()
        df.loc[label_indices, splited_labels] = 1

    return df


# =============================
# 读取数据 & one-hot
# =============================

train_df = read_image_labels(TRAIN_DATA_FILE).sample(frac=1.0, random_state=42)
print(f"Total training samples: {len(train_df)}")

tr_df = get_one_hot_encoded_labels(train_df)

# 6 个基础类别（多标签输出维度）
CLASSES = [
    "rust",
    "complex",
    "healthy",
    "powdery_mildew",
    "scab",
    "frog_eye_leaf_spot",
]

# =============================
# Train / Val 划分
# =============================

X_train, X_valid, Y_train, Y_valid = train_test_split(
    pd.Series(train_df.index),
    np.array(tr_df[CLASSES]),
    test_size=0.2,
    random_state=42,
)

print(f"Train size: {len(X_train)}, Valid size: {len(X_valid)}")


# =============================
# 图像加载 & Dataset
# =============================

folders = dict(
    {
        "data": DATA_ROOT,
        "train": TRAIN_DIR,
        "val": TRAIN_DIR,   # 验证集使用同一目录，只是索引不同
        "test": TEST_DIR,
    }
)


def get_image(image_id, kind: str = "train") -> Image.Image:
    """Load an image from file."""
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)


class PlantDataset(Dataset):
    def __init__(
        self,
        image_ids: pd.Series,
        targets: np.ndarray,
        transform=None,
        target_transform=None,
        kind: str = "train",
    ):
        self.image_ids = image_ids.reset_index(drop=True)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.kind = kind

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # load and transform image
        img = np.array(get_image(self.image_ids.iloc[idx], kind=self.kind))

        if self.transform:
            img = self.transform(image=img)["image"]

        # target
        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target


# =============================
# Albumentations transforms
# =============================

train_transform = A.Compose(
    [
        A.RandomResizedCrop(height=IM_SIZE, width=IM_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(height=IM_SIZE, width=IM_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# =============================
# Dataloaders
# =============================

trainset = PlantDataset(X_train, Y_train, transform=train_transform, kind="train")
validset = PlantDataset(X_valid, Y_valid, transform=val_transform, kind="val")

trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=4)
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False, num_workers=4)


# =============================
# 模型定义：Inception v3 多标签
# =============================

model = torchvision.models.inception_v3(pretrained=True)
model.aux_logits = False
model.fc = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(2048, len(CLASSES)),
    nn.Sigmoid(),
)
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =============================
# 训练监控工具
# =============================

class MetricMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.scores = []
        self.metrics = dict({"loss": self.losses, "f1": self.scores})

    def update(self, metric_name, value):
        self.metrics[metric_name].append(value)


monitor = MetricMonitor()


# =============================
# 训练主循环
# =============================

def train(num_epochs: int = 20, threshold: float = 0.4):
    os.makedirs(BESTMODEL_DIR, exist_ok=True)

    best_f1 = 0.0

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        train_batches = 0

        train_metric = BinaryF1Score(threshold=threshold).to(DEVICE)

        for images, labels in trainloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images.float())
            loss = criterion(preds.float(), labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            train_f1 += train_metric(preds, labels).item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_f1 = train_f1 / max(train_batches, 1)

        # ---------- Valid ----------
        model.eval()
        valid_loss = 0.0
        valid_f1 = 0.0
        valid_batches = 0

        valid_metric = BinaryF1Score(threshold=threshold).to(DEVICE)

        with torch.no_grad():
            for images, labels in validloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = model(images.float())
                loss = criterion(preds.float(), labels.float())

                valid_loss += loss.detach().item()
                valid_f1 += valid_metric(preds, labels).item()
                valid_batches += 1

        avg_valid_loss = valid_loss / max(valid_batches, 1)
        avg_valid_f1 = valid_f1 / max(valid_batches, 1)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"| Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f} "
            f"| Valid Loss: {avg_valid_loss:.4f}, Valid F1: {avg_valid_f1:.4f}"
        )

        monitor.update("loss", avg_valid_loss)
        monitor.update("f1", avg_valid_f1)

        # ---------- Save best model ----------
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }

        if avg_valid_f1 > best_f1:
            best_f1 = avg_valid_f1
            best_path = os.path.join(
                BESTMODEL_DIR,
                f"inception_v3_bestmodel_epoch{epoch+1}.pth",
            )
            torch.save(checkpoint, best_path)
            print(f"  -> New best model saved to: {best_path}")

        # ---------- Optional: 每 N 个 epoch 额外存一份 ----------
        if (epoch + 1) % 20 == 0:
            epoch_path = os.path.join(
                BESTMODEL_DIR,
                f"inception_v3_epoch{epoch+1}.pth",
            )
            torch.save(checkpoint, epoch_path)
            print(f"  -> Checkpoint saved to: {epoch_path}")


if __name__ == "__main__":
    train(num_epochs=20, threshold=0.4)
