# -*- coding: utf-8 -*-
"""FashionIQ 小 Demo 后端 —— 点选参考图 + 中文指令 → 纯 CLIP 融合检索。

设计要点（刻意保持轻量，只在 WSL conda `project_env` 里跑）：
- 图库 & 参考图向量都直接复用离线预计算好的 CLIP npy，
  因此参考图无需再编码；每个请求只需实时编码 1 条中文翻译后的英文查询。
- 查询 = normalize(ref_vec + alpha * text_vec)，与全库向量点积排序，排除参考图本身。
- 不依赖 Ray / vLLM / LLM，CLIP 文本编码器懒加载（cuda 优先）。

运行（在仓库根目录下执行）：
  conda activate project_env
  cd src/demo_fiq
  python -m uvicorn server:app --host 0.0.0.0 --port 8899
浏览器打开 http://127.0.0.1:8899
"""
import json
import mimetypes
import os
import random
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------- 路径（demo 目录固定位于 src/demo_fiq） ----------------
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SELF_DIR, "..", ".."))          # 仓库根目录
# CLIP 权重：可用环境变量 CLIP_PATH 指向本地目录，默认回退到 HuggingFace 模型 ID
CLIP_PATH = os.environ.get("CLIP_PATH", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
STATIC_DIR = os.path.join(SELF_DIR, "static")
ASSETS_DIR = os.path.join(SELF_DIR, "assets")      # 演示案例等产物
ALPHA_DEFAULT = 0.7      # ref 向量与 text 向量的融合权重
TOP_K_DEFAULT = 8

# 图库：真实 FashionIQ 服装商品图。npy 由 tools/fiq_build_db.py 生成
# （data/FashionIQ_real/clip_db/...），只需 image 向量，无需 caption 向量。
IMG_DIR = os.path.join(ROOT, "data", "FashionIQ_real", "images")
NPY_DIR = os.path.join(ROOT, "data", "FashionIQ_real", "clip_db", "CLIP-ViT-B-32")
DB_LABEL = "FashionIQ 真实服装图"
GALLERY_LIMIT = 240

IMG_EXTS = [".jpg", ".jpeg", ".png"]   # 图名(无扩展名) → 磁盘文件，宽容几种扩展名

# ---------------- 图库：npy 已算好，只读加载 ----------------
def img_path(name):
    """图名 → 磁盘文件路径（宽容扩展名），找不到返回 None。"""
    for ext in IMG_EXTS:
        p = os.path.join(IMG_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None


def _load_image_db():
    image_name = np.load(os.path.join(NPY_DIR, "image_name_list.npy"), allow_pickle=True)
    image_name = [str(x) for x in image_name.tolist()]
    image_emb = np.load(os.path.join(NPY_DIR, "image_embedding.npy")).astype(np.float32)
    # 磁盘上未归一化，这里统一 L2 归一化
    norms = np.linalg.norm(image_emb, axis=1, keepdims=True)
    image_emb = image_emb / np.maximum(norms, 1e-9)
    # 只保留文件真实存在、且向量能对齐的图
    kept_name, kept_vec = [], []
    for name, vec in zip(image_name, image_emb):
        if img_path(name) is not None:
            kept_name.append(name)
            kept_vec.append(vec)
    if not kept_name:
        raise RuntimeError(f"图库为空，请检查 {IMG_DIR} 与 {NPY_DIR}")
    return kept_name, np.stack(kept_vec, axis=0)

IMAGE_NAMES, IMAGE_EMB = _load_image_db()
NAME2IDX = {n: i for i, n in enumerate(IMAGE_NAMES)}

# 真实库上万张，前端网格没必要全展示：固定随机顺序 + 分批窗口浏览
_rng = random.Random(20260908)
GALLERY_ORDER = list(range(len(IMAGE_NAMES)))
_rng.shuffle(GALLERY_ORDER)


def _ensure_valid_name(name):
    if name not in NAME2IDX:
        raise HTTPException(status_code=404, detail=f"unknown image: {name}")
    return NAME2IDX[name]


# ---------------- CLIP 文本编码器（懒加载、只编码文本） ----------------
_clip_lock = threading.Lock()
_clip = None
_proc = None
_device = None


def ensure_clip():
    global _clip, _proc, _device
    if _clip is None:
        with _clip_lock:
            if _clip is None:
                import torch
                from transformers import AutoProcessor, CLIPModel

                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _clip = CLIPModel.from_pretrained(CLIP_PATH).to(_device).eval()
                _proc = AutoProcessor.from_pretrained(CLIP_PATH)
    return _clip, _proc, _device


def encode_text(text):
    """单条英文文本 -> L2 归一化向量 (1, dim) float32 numpy。"""
    import torch

    model, proc, dev = ensure_clip()
    inputs = proc.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = torch.nn.functional.normalize(emb, dim=1).float()
    return emb.detach().cpu().numpy()


# ---------------- 中 → 英 迷你翻译（够演示即可） ----------------
COLOR_TABLE = [
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow"),
    ("紫色", "purple"), ("粉色", "pink"), ("橙色", "orange"), ("棕色", "brown"),
    ("灰色", "gray"), ("黑色", "black"), ("白色", "white"), ("青色", "cyan"),
    ("大红", "red"), ("深蓝", "dark blue"), ("浅蓝", "light blue"),
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("黄", "yellow"),
    ("紫", "purple"), ("粉", "pink"), ("橙", "orange"), ("棕", "brown"),
    ("灰", "gray"), ("黑", "black"), ("白", "white"),
]
SHAPE_TABLE = [
    ("圆形", "circle"), ("椭圆形", "ellipse"), ("三角形", "triangle"),
    ("长方形", "rectangle"), ("正方形", "square"), ("椭圆", "ellipse"),
    ("圆", "circle"), ("三角", "triangle"), ("方形", "square"),
    ("方块", "square"), ("矩形", "rectangle"), ("线条", "line"),
    ("线", "line"),
]
# 服装/物件名词（真图库上用得上；长词在前）
NOUN_TABLE = [
    ("衣服", "outfit"), ("服装", "outfit"), ("连衣裙", "dress"), ("长裙", "dress"),
    ("短裙", "skirt"), ("裙子", "skirt"),
    ("晚礼服", "evening gown"), ("T恤", "t-shirt"), ("T 恤", "t-shirt"),
    ("长袖衬衫", "long sleeve shirt"), ("衬衫", "shirt"), ("衬衣", "shirt"),
    ("卫衣", "hoodie"), ("毛衣", "sweater"), ("开衫", "cardigan"),
    ("西装外套", "blazer"), ("外套", "jacket"), ("夹克", "jacket"), ("大衣", "coat"),
    ("牛仔外套", "denim jacket"), ("上衣", "top"), ("t恤", "t-shirt"),
    ("裤子", "pants"), ("长裤", "trousers"), ("牛仔裤", "jeans"),
    ("短裤", "shorts"), ("运动裤", "sweatpants"), ("半身裙", "skirt"),
    ("鞋子", "shoes"), ("鞋", "shoes"), ("靴子", "boots"),
    ("包包", "handbag"), ("帽子", "hat"), ("围巾", "scarf"),
]
# 表示"去掉/不要某颜色"的反义动作 → 不套改色模板，退回原文查询
NEGATIVE_VERBS = ["去掉", "删除", "移除", "不要", "减少"]
ACTIONS = ["改成", "变成", "变为", "换成", "染成", "换"]


def _first_of(text, table):
    """返回 text 中按表顺序命中的第一个中文词及其英文（长词优先已由表序保证）。"""
    if not text:
        return None, None
    for cn, en in table:
        if cn in text:
            return cn, en
    return None, None


def _mapped_target(s, right):
    """从改后片段/全句里抽 颜色 + (服装名词或形状)，拼成目标短语。"""
    parts = []
    _, color_en = _first_of(right, COLOR_TABLE)
    if not color_en:
        _, color_en = _first_of(s, COLOR_TABLE)
    _, noun_en = _first_of(right, NOUN_TABLE)
    if not noun_en:
        _, noun_en = _first_of(s, NOUN_TABLE)
    if not noun_en:
        _, noun_en = _first_of(s, SHAPE_TABLE)   # 形状兜底
    if color_en:
        parts.append(color_en)
    if noun_en:
        parts.append(noun_en)
    if parts:
        return "a " + " ".join(parts)
    return None


def translate_instruction(inst_cn):
    """中文修改指令 → 英文 CLIP 查询。

    返回 (query_en, mapped)：mapped=True 表示成功识别「把 X 改成 Y」类改色/换装语义；
    解析不到或属「去掉…」反义语义时退回原文（query_en=inst_cn, mapped=False）。
    """
    s = (inst_cn or "").strip()
    if not s:
        return "", False
    if any(v in s for v in NEGATIVE_VERBS):
        return s, False
    action = None
    for act in ACTIONS:
        if act in s:
            left, right = s.split(act, 1)
            action = act
            break
    right = right if action else s
    q = _mapped_target(s, right)
    if q:
        return q, True
    return s, False


# ---------------- 检索 ----------------
def search(ref_name, inst_cn, top_k=TOP_K_DEFAULT, alpha=ALPHA_DEFAULT):
    """融合检索：返回 dict(names, scores, query_en, mapped, ms)。"""
    t0 = time.time()
    ref_idx = _ensure_valid_name(ref_name)
    query_en, mapped = translate_instruction(inst_cn)
    if not query_en.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    text_vec = encode_text(query_en)[0].astype(np.float32)   # (dim,)
    ref_vec = IMAGE_EMB[ref_idx]
    fused = ref_vec + float(alpha) * text_vec
    fused = fused / max(np.linalg.norm(fused), 1e-9)

    sim = IMAGE_EMB @ fused          # 全库点积 (n,)
    order = np.argsort(-sim)
    order = order[order != ref_idx]  # 排除参考图自身
    order = order[:top_k]

    names = [IMAGE_NAMES[i] for i in order]
    scores = [round(float(sim[i]), 4) for i in order]
    return {
        "ref": ref_name,
        "inst_cn": inst_cn,
        "query_en": query_en,
        "mapped": mapped,
        "alpha": alpha,
        "names": names,
        "scores": scores,
        "urls": [f"/img/{n}" for n in names],
        "ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------- FastAPI ----------------
# 推荐演示案例（由 find_cases.py 生成 cases.json）
CASES_PATH = os.path.join(ASSETS_DIR, "cases.json")


def load_cases():
    try:
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


class SearchBody(BaseModel):
    name: str
    inst_cn: str
    top_k: int = TOP_K_DEFAULT
    alpha: float | None = None


app = FastAPI(title="FashionIQ 小 Demo")


@app.get("/api/images")
def api_images(batch: int = 0):
    """按批返回图库：库大时前端分批浏览（固定随机顺序，batch=0/1/2…）。"""
    n = len(IMAGE_NAMES)
    lim = max(1, min(GALLERY_LIMIT, n))
    if batch < 0:
        batch = 0
    start = batch * lim
    idxs = GALLERY_ORDER[start:start + lim]
    total_batches = (n + lim - 1) // lim
    return {
        "images": [{"name": IMAGE_NAMES[i], "url": f"/img/{IMAGE_NAMES[i]}"} for i in idxs],
        "total": n,
        "shown": len(idxs),
        "batch": batch,
        "has_more": (batch + 1) < total_batches,
        "total_batches": total_batches,
        "db": DB_LABEL,
    }


@app.get("/img/{name}")
def image_file(name: str):
    _ensure_valid_name(name)
    path = img_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="file missing on disk")
    media = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media)


@app.get("/api/cases")
def api_cases():
    return {"cases": load_cases()}


@app.post("/api/search")
def api_search(body: SearchBody):
    alpha = ALPHA_DEFAULT if body.alpha is None else float(body.alpha)
    return search(body.name, body.inst_cn, top_k=int(body.top_k), alpha=alpha)


# 静态前端放最后挂载（html=True 自动提供 index.html）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8899)))
