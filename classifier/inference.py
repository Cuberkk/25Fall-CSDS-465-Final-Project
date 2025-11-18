#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.classification import BinaryF1Score


# =============================
# 基本配置
# =============================

BATCH = 16
LR = 1e-4
IM_SIZE = 299

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./plant-pathology-2021-fgvc8"
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DATA_FILE = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

CHECKPOINT_PATH = os.path.join(
    DATA_ROOT, "inception_v3_bestmodel", "inception_v3_bestmodel_epoch20.pth"
)

CLASSES = [
    "rust",
    "complex",
    "healthy",
    "powdery_mildew",
    "scab",
    "frog_eye_leaf_spot",
]


# =============================
# 读取 & 处理标签
# =============================

def read_image_labels() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_DATA_FILE).set_index("image")
    return df


def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes."""
    single_labels = []
    for label in unique_labels:
        single_labels += label.split()
    single_labels = set(single_labels)
    return list(single_labels)


def get_one_hot_encoded_labels(dataset_df: pd.DataFrame) -> pd.DataFrame:
    df = dataset_df.copy()

    unique_labels = df.labels.unique()
    column_names = get_single_labels(unique_labels)

    df[column_names] = 0

    # one-hot / multi-hot
    for label in unique_labels:
        label_indices = df[df["labels"] == label].index
        splited_labels = label.split()
        df.loc[label_indices, splited_labels] = 1

    return df


# =============================
# Dataset & transforms
# =============================

folders = dict(
    {
        "data": DATA_ROOT,
        "train": TRAIN_DIR,
        "val": TRAIN_DIR,
        "test": TEST_DIR,
    }
)


def get_image(image_id: str, kind: str = "train") -> Image.Image:
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)


class PlantDataset(Dataset):
    def __init__(
        self,
        image_ids,
        targets,
        transform=None,
        target_transform=None,
        kind="train",
    ):
        self.image_ids = image_ids.reset_index(drop=True) if isinstance(
            image_ids, pd.Series
        ) else image_ids
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.kind = kind

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # image_id 可能是 Series，也可能是 list / numpy
        if isinstance(self.image_ids, pd.Series):
            image_id = self.image_ids.iloc[idx]
        else:
            image_id = self.image_ids[idx]

        img = np.array(get_image(image_id, kind=self.kind))

        if self.transform:
            img = self.transform(image=img)["image"]

        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target


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
# 模型定义 & 加载
# =============================

def build_model() -> nn.Module:
    model = torchvision.models.inception_v3(pretrained=True)
    model.aux_logits = False
    # 注意：这里是 Linear(2048 -> 6) + Sigmoid，和你原来 inference 代码一致
    model.fc = nn.Sequential(
        nn.Linear(2048, 6),
        nn.Sigmoid(),
    )
    return model


def load_model(checkpoint_path: str) -> nn.Module:
    model = build_model().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# =============================
# 评估：获取 y_true & y_pred_proba
# =============================

def evaluate_on_valid(model: nn.Module, validloader: DataLoader):
    y_true = np.empty(shape=(0, len(CLASSES)), dtype=int)
    y_pred_proba = np.empty(shape=(0, len(CLASSES)), dtype=float)

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(validloader):
            X = X.to(DEVICE)
            y = y.numpy()  # labels 本身已经是 numpy 数组

            pred = model(X).detach().cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred_proba = np.vstack((y_pred_proba, pred))

    return y_true, y_pred_proba


def plot_confusion_matrix(
    y_test,
    y_pred_proba,
    threshold: float = 0.4,
    label_names=CLASSES,
) -> None:
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    c_matrices = multilabel_confusion_matrix(y_test, y_pred)

    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    for cm, label, ax in zip(c_matrices, label_names, axes.flatten()):
        sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap=cmap)

        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"{label}")

    plt.tight_layout()
    plt.show()


def compute_f1(y_true, y_pred_proba, threshold: float = 0.4) -> float:
    f1_metric = BinaryF1Score(threshold=threshold)
    y_pred = torch.as_tensor((y_pred_proba > threshold).astype(int))
    y_true_t = torch.as_tensor(y_true.astype(int))
    f1 = f1_metric(y_pred, y_true_t).item()
    return f1


# =============================
# 生成 Kaggle submission
# =============================

def save_submission(model: nn.Module, threshold: float = 0.4):
    """
    对 test_images 生成预测，并按照多标签格式生成 submission.csv
    """
    image_ids_df = pd.read_csv(SAMPLE_SUB_PATH)  # 有 image 和 labels 两列

    # 构造 dummy target（全 0），实际不会用到
    dummy_targets = np.zeros((len(image_ids_df), len(CLASSES)), dtype=np.float32)

    dataset = PlantDataset(
        image_ids=image_ids_df["image"],
        targets=dummy_targets,
        transform=val_transform,
        kind="test",
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

    model.eval()
    all_pred_labels = []

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(DEVICE).float()
            y_pred_proba = model(X).detach().cpu().numpy()

            # 多标签阈值化
            y_pred_bin = (y_pred_proba > threshold).astype(int)

            for row in y_pred_bin:
                # 找到所有为 1 的类别
                idxs = np.where(row == 1)[0]
                if len(idxs) == 0:
                    # 如果没有任何类别超过阈值，可以默认 healthy 或空
                    pred_labels = "healthy"
                else:
                    pred_labels = " ".join([CLASSES[i] for i in idxs])
                all_pred_labels.append(pred_labels)

    image_ids_df["labels"] = all_pred_labels
    image_ids_df.set_index("image", inplace=True)
    out_path = os.path.join("./", "submission.csv")
    image_ids_df.to_csv(out_path)
    print(f"Submission saved to: {out_path}")
    return image_ids_df


# =============================
# main 流程
# =============================

def main():
    # 1) 准备数据 & valid loader
    train_df = read_image_labels().sample(frac=1.0, random_state=42)
    tr_df = get_one_hot_encoded_labels(train_df)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        pd.Series(train_df.index),
        np.array(tr_df[CLASSES]),
        test_size=0.2,
        random_state=42,
    )

    validset = PlantDataset(
        X_valid,
        Y_valid,
        transform=val_transform,
        kind="val",
    )
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    # 2) 加载模型
    model = load_model(CHECKPOINT_PATH)

    # 3) 在验证集上评估
    y_true, y_pred_proba = evaluate_on_valid(model, validloader)
    plot_confusion_matrix(y_true, y_pred_proba, threshold=0.4)

    f1 = compute_f1(y_true, y_pred_proba, threshold=0.4)
    print(f"Valid Binary F1 (global): {f1:.4f}")

    f1_df = pd.DataFrame({"name": ["F1"], "score": [f1]}).set_index("name")
    print(f1_df)

    # 4) 生成 submission.csv
    save_submission(model, threshold=0.4)


if __name__ == "__main__":
    main()
