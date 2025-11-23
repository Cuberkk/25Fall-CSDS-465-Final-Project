"""
Test LeafGAN on a specific folder and save generated disease images.
"""

import os
from pathlib import Path

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im, save_image 


if __name__ == '__main__':
    opt = TestOptions().parse()

    # ---------- hard-code test params ----------
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.dataset_mode = "single"

    # ---------- create dataset & model ----------
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    # ---------- output dir ----------
    DATA_ROOT = "./data"
    # 例：results/leafgan_run/test_latest/images_only/
    out_dir = Path("/home/erie_lab/Documents/kxz365/ECSE465/25Fall-CSDS-465-Final-Project/data/generated_complex")
    if out_dir.exists():
        os.remove(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving generated images to: {out_dir}")

    # ---------- inference ----------
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_paths = model.get_image_paths()

        # ---------------------------------------------------------
        # ✅ 这里取生成图：
        # CycleGAN/test 通常是 visuals['fake_B']
        # 但有的实现叫 'fake' / 'fake_B' / 'generated'
        # 你只要把下面这一行 key 改成你实际的即可
        # ---------------------------------------------------------
        fake = visuals.get('fake_B', None)
        if fake is None:
            fake = visuals.get('fake', None)
        if fake is None:
            raise KeyError(
                f"Cannot find generated image in visuals keys: {visuals.keys()}"
            )

        fake_img = tensor2im(fake)   # tensor(H,W,C) -> uint8 numpy

        # 取原始文件名
        src_path = Path(img_paths[0])
        save_name = src_path.stem + "_complex.png"
        save_path = out_dir / save_name

        save_image(fake_img, str(save_path), aspect_ratio=opt.aspect_ratio)

        if i % 20 == 0:
            print(f"processed {i:04d}: {src_path.name} -> {save_name}")

    print("[DONE] All images generated.")