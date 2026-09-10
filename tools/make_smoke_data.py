"""Generate a minimal, structurally-valid FashionIQ-dress smoke dataset.

Purpose: end-to-end exercise of the CIR pipeline (image_db build -> Stage1
reasoner -> CLIP retrieval -> Stage2 verifier -> Stage3 evaluator) on a small
GPU, WITHOUT downloading the real 30k-image FashionIQ set. The 40 images are
synthetic placeholders, so retrieval metrics are meaningless here.
"""
import json
import os
import random
from PIL import Image, ImageDraw

random.seed(0)

# 输出根目录：可用环境变量 FIQ_DATA 覆盖，默认 <仓库根>/data/FashionIQ
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("FIQ_DATA", os.path.join(_REPO_ROOT, "data", "FashionIQ"))
IMG_DIR = os.path.join(ROOT, "images")
SPLIT_DIR = os.path.join(ROOT, "image_splits")
CAP_DIR = os.path.join(ROOT, "captions")
for d in (IMG_DIR, SPLIT_DIR, CAP_DIR):
    os.makedirs(d, exist_ok=True)

M = 40  # images in the retrieval pool
names = [f"smk_{i:05d}" for i in range(M)]

# Distinct synthetic images (colored background + simple shapes) so PIL / CLIP
# / the captioner all have something real to process.
for i, name in enumerate(names):
    w, h = 256, 384
    img = Image.new("RGB", (w, h), (random.randint(40, 215),) * 3)
    dr = ImageDraw.Draw(img)
    for _ in range(4):
        shape = random.choice(["rect", "ellipse", "line"])
        x0, y0 = random.randint(0, w // 2), random.randint(0, h // 2)
        x1, y1 = random.randint(w // 2, w), random.randint(h // 2, h)
        fill = tuple(random.randint(0, 255) for _ in range(3))
        if shape == "rect":
            dr.rectangle([x0, y0, x1, y1], fill=fill)
        elif shape == "ellipse":
            dr.ellipse([x0, y0, x1, y1], fill=fill)
        else:
            dr.line([x0, y0, x1, y1], fill=fill, width=6)
    img.save(os.path.join(IMG_DIR, f"{name}.png"))

# split.dress.val.json : plain list of image names
with open(os.path.join(SPLIT_DIR, "split.dress.val.json"), "w") as f:
    json.dump(names, f)

# cap.dress.val.json : official item format {candidate, target, captions:[..]}
pairs = [
    ("smk_00003", "smk_00017", ["make it shorter", "make the dress black"]),
    ("smk_00010", "smk_00025", ["add long sleeves", "turn it into a casual look"]),
    ("smk_00022", "smk_00038", ["make it floor length", "use a floral pattern"]),
]
captions = [
    {"candidate": c, "target": t, "captions": caps} for c, t, caps in pairs
]
with open(os.path.join(CAP_DIR, "cap.dress.val.json"), "w") as f:
    json.dump(captions, f)

print(f"wrote {M} images + {len(captions)} queries to {ROOT}")
