# -*- coding: utf-8 -*-
"""把 FashionIQ val parquet 里的图片字节解出来，落到 images/<name>.jpg。

- 列：category/split/id/img（img 是 {'bytes': jpeg_bytes}）
- 文件扩展名统一 .jpg（源本身是 JPEG）
- 121 个 id 重复（同款不同图）→ 用 id + `__k` 后缀去重，保证名字全局唯一
- 附带写 meta.jsonl {name, id, category}，供后续按需引用

输出目录默认为 <仓库根>/data/FashionIQ_real，可用环境变量 FIQ_REAL_DIR 覆盖。
用法（需 pyarrow）：python tools/fiq_decode.py
"""
import json
import os
import re

import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("FIQ_REAL_DIR", os.path.join(REPO_ROOT, "data", "FashionIQ_real"))
PARQUET = os.path.join(OUT_ROOT, "val-00000-of-00001.parquet")
IMG_DIR = os.path.join(OUT_ROOT, "images")
META = os.path.join(OUT_ROOT, "meta.jsonl")
SAFE = re.compile(r"[^A-Za-z0-9_.\-]")


def grab(cell):
    if isinstance(cell, dict):
        b = cell.get(b"bytes", cell.get("bytes"))
        return bytes(b) if b is not None else None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = pq.ParquetFile(PARQUET).read().to_pandas()
    print("rows:", len(df), flush=True)
    seen, wrote, skipped = {}, 0, 0
    with open(META, "w", encoding="utf-8") as mf:
        for i, row in enumerate(df.itertuples(index=False)):
            bid = grab(row.img)
            if not bid or len(bid) < 2:
                skipped += 1
                continue
            base = SAFE.sub("_", str(row.id)) or f"img_{i}"
            k = seen.get(base, 0)
            name = base if k == 0 else f"{base}__{k}"
            seen[base] = k + 1
            with open(os.path.join(IMG_DIR, name + ".jpg"), "wb") as f:
                f.write(bid)
            mf.write(json.dumps({"name": name, "id": str(row.id),
                                 "category": str(row.category)},
                                ensure_ascii=False) + "\n")
            wrote += 1
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(df)} wrote={wrote}", flush=True)
    print(f"DONE wrote={wrote} skipped={skipped} -> {IMG_DIR}", flush=True)


if __name__ == "__main__":
    main()
