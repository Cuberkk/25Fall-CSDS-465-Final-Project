#!/usr/bin/env python
# coding: utf-8

"""
Prepare LeafGAN dataset directories from Plant Pathology 2021 (FGVC8).

For a given target class (e.g. "rust"), this script will:

1. Read train.csv under --pp-root.
2. Collect:
   - All images whose labels contain "healthy"  -> domain A
   - All images whose labels contain target_class -> domain B
3. Split each list into train / test by --test-ratio.
4. Create the following structure under --out-root:

   <out_root>/healthy2<target_class>/
       trainA/   (healthy train images)
       testA/    (healthy test images)
       trainB/   (target_class train images)
       testB/    (target_class test images)

By default it copies images. Use --copy-mode symlink to create symlinks instead.

Example:

    python prepare_leafgan_data.py \
        --pp-root ./plant-pathology-2021-fgvc8 \
        --out-root ./leafgan_data \
        --target-class rust \
        --test-ratio 0.1

"""

import argparse
import os
import shutil
import random
from typing import List, Tuple

import pandas as pd


# 六个基础类别（与训练脚本中的 CLASSES 一致）
CLASSES = [
    "rust",
    "complex",
    "healthy",
    "powdery_mildew",
    "scab",
    "frog_eye_leaf_spot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LeafGAN-style dataset from Plant Pathology 2021."
    )
    parser.add_argument(
        "--pp-root",
        type=str,
        default="./plant-pathology-2021-fgvc8",
        help="Root directory of Plant Pathology dataset (contains train.csv, train_images/).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./leafgan_data",
        help="Root directory to store LeafGAN datasets.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        required=True,
        choices=CLASSES,
        help="Target basic class to form domain B (e.g. rust, complex, scab...).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of images to use for testA/testB (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting train/test.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="How to place images in LeafGAN folders: copy or symlink.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, without creating dirs or copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing output directory for this target_class before writing.",
    )

    args = parser.parse_args()
    return args


def read_train_df(pp_root: str) -> pd.DataFrame:
    csv_path = os.path.join(pp_root, "train.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"train.csv not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "labels" not in df.columns:
        raise ValueError("train.csv must contain 'image' and 'labels' columns.")
    return df


def collect_image_lists(df: pd.DataFrame, target_class: str) -> Tuple[List[str], List[str]]:
    """Return healthy_images, target_images from train.csv dataframe."""
    def has_token(label: str, token: str) -> bool:
        parts = str(label).split()
        return token in parts

    healthy_mask = df["labels"].apply(lambda s: has_token(s, "healthy"))
    target_mask = df["labels"].apply(lambda s: has_token(s, target_class))

    healthy_imgs = df.loc[healthy_mask, "image"].tolist()
    target_imgs = df.loc[target_mask, "image"].tolist()

    return healthy_imgs, target_imgs


def train_test_split_list(
    names: List[str],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    names = list(names)
    random.Random(seed).shuffle(names)
    if not names:
        return [], []
    n_test = max(1, int(len(names) * test_ratio))
    test_list = names[:n_test]
    train_list = names[n_test:]
    return train_list, test_list


def prepare_dirs(
    out_root: str,
    target_class: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, str, str, str]:
    """Create (or check) LeafGAN directories and return their paths."""
    dataset_root = os.path.join(out_root, f"healthy2{target_class}")
    trainA = os.path.join(dataset_root, "trainA")
    testA = os.path.join(dataset_root, "testA")
    trainB = os.path.join(dataset_root, "trainB")
    testB = os.path.join(dataset_root, "testB")

    if os.path.exists(dataset_root):
        if overwrite:
            print(f"[INFO] Removing existing directory: {dataset_root}")
            if not dry_run:
                shutil.rmtree(dataset_root)
        else:
            raise FileExistsError(
                f"Output directory already exists: {dataset_root} "
                f"(use --overwrite to remove and recreate)"
            )

    print(f"[INFO] Creating LeafGAN dataset at: {dataset_root}")
    if not dry_run:
        os.makedirs(trainA, exist_ok=True)
        os.makedirs(testA, exist_ok=True)
        os.makedirs(trainB, exist_ok=True)
        os.makedirs(testB, exist_ok=True)

    return trainA, testA, trainB, testB


def place_images(
    img_root: str,
    img_names: List[str],
    dst_dir: str,
    mode: str = "copy",
    dry_run: bool = False,
) -> None:
    assert mode in ["copy", "symlink"]
    if not img_names:
        return
    for name in img_names:
        src = os.path.join(img_root, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.isfile(src):
            print(f"[WARN] Source image not found, skip: {src}")
            continue
        if dry_run:
            print(f"[DRY-RUN] {mode} {src} -> {dst}")
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            # 如果已存在旧链接/文件，先删掉
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)


def main():
    args = parse_args()

    print("==== prepare_leafgan_data.py ====")
    print(f"pp-root      : {args.pp_root}")
    print(f"out-root     : {args.out_root}")
    print(f"target-class : {args.target_class}")
    print(f"test-ratio   : {args.test_ratio}")
    print(f"copy-mode    : {args.copy_mode}")
    print(f"dry-run      : {args.dry_run}")
    print(f"overwrite    : {args.overwrite}")
    print("=================================")

    # 1) 读取 train.csv
    df = read_train_df(args.pp_root)

    # 2) 收集 healthy & target_class 图像列表
    healthy_imgs, target_imgs = collect_image_lists(df, args.target_class)

    print(f"[INFO] Found {len(healthy_imgs)} healthy images.")
    print(f"[INFO] Found {len(target_imgs)} images containing '{args.target_class}'.")

    # 3) train/test 划分
    trainA_list, testA_list = train_test_split_list(
        healthy_imgs, args.test_ratio, args.seed
    )
    trainB_list, testB_list = train_test_split_list(
        target_imgs, args.test_ratio, args.seed
    )

    print(f"[INFO] healthy  -> trainA: {len(trainA_list)}, testA: {len(testA_list)}")
    print(f"[INFO] {args.target_class} -> trainB: {len(trainB_list)}, testB: {len(testB_list)}")

    if args.dry_run:
        print("[DRY-RUN] No directories will be created, no files will be copied.")
    # 4) 创建目录
    trainA_dir, testA_dir, trainB_dir, testB_dir = prepare_dirs(
        args.out_root, args.target_class, args.overwrite, args.dry_run
    )

    # 5) 拷贝 / 链接图片
    img_root = os.path.join(args.pp_root, "train_images")
    if not os.path.isdir(img_root):
        raise NotADirectoryError(f"train_images directory not found at: {img_root}")

    print("[INFO] Placing healthy images into trainA/testA ...")
    place_images(img_root, trainA_list, trainA_dir, mode=args.copy_mode, dry_run=args.dry_run)
    place_images(img_root, testA_list, testA_dir, mode=args.copy_mode, dry_run=args.dry_run)

    print(f"[INFO] Placing '{args.target_class}' images into trainB/testB ...")
    place_images(img_root, trainB_list, trainB_dir, mode=args.copy_mode, dry_run=args.dry_run)
    place_images(img_root, testB_list, testB_dir, mode=args.copy_mode, dry_run=args.dry_run)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
