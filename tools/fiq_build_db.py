# -*- coding: utf-8 -*-
"""对真实服装图库跑 laion ViT-B/32 图像编码，生成 image_db（仅 image 向量，无需 caption）。

输出：clip_db/CLIP-ViT-B-32/{image_name_list,image_embedding}.npy
  image_name_list.npy  = 图名(无扩展名, 排序稳定)
  image_embedding.npy  = (N,512) float32，未归一化（server 载入后会再 L2 归一化）

输出目录默认为 <仓库根>/data/FashionIQ_real，可用环境变量 FIQ_REAL_DIR 覆盖。
CLIP 权重可用环境变量 CLIP_PATH 指向本地目录，默认走 HuggingFace 模型 ID。
用法（需 torch/transformers，建议 GPU）：python tools/fiq_build_db.py
"""
import glob
import os
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("FIQ_REAL_DIR", os.path.join(REPO_ROOT, "data", "FashionIQ_real"))
IMG_DIR = os.path.join(OUT_ROOT, "images")
OUT_DIR = os.path.join(OUT_ROOT, "clip_db", "CLIP-ViT-B-32")
CLIP_PATH = os.environ.get("CLIP_PATH", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
BATCH = 128


def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"images: {len(files)}", flush=True)

    import torch
    from transformers import AutoProcessor, CLIPModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    model = CLIPModel.from_pretrained(CLIP_PATH)
    model = model.half().to(dev).eval()
    proc = AutoProcessor.from_pretrained(CLIP_PATH)

    names = [os.path.splitext(os.path.basename(p))[0] for p in files]
    emb = np.zeros((len(files), model.config.projection_dim), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(files), BATCH):
            chunk = files[s:s + BATCH]
            from PIL import Image
            pils = [Image.open(p).convert("RGB") for p in chunk]
            px = proc(images=pils, return_tensors="pt").pixel_values.half().to(dev)
            v = model.get_image_features(pixel_values=px)      # (b, dim) fp16
            emb[s:s + len(chunk)] = v.float().cpu().numpy()
            if (s // BATCH) % 8 == 0 or s + BATCH >= len(files):
                print(f"  {min(s+BATCH, len(files))}/{len(files)}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "image_name_list.npy"), np.array(names, dtype=object))
    np.save(os.path.join(OUT_DIR, "image_embedding.npy"), emb)
    print(f"DONE  {len(files)}x{emb.shape[1]}  {(time.time()-t0)/60:.1f}m  -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
