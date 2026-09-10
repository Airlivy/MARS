# -*- coding: utf-8 -*-
"""Resumable segmented download of the FashionIQ val image parquet from hf-mirror.

Only stdlib (urllib). Idempotent: re-running skips finished segments.

从 hf-mirror 分片多线程下载 FashionIQ val 图像 parquet（约 207MB），
支持断点续传：重跑会跳过已完成的分片。

输出目录默认为 <仓库根>/data/FashionIQ_real，可用环境变量 FIQ_REAL_DIR 覆盖。
用法：python tools/fiq_download.py
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("FIQ_REAL_DIR", os.path.join(REPO_ROOT, "data", "FashionIQ_real"))
URL = ("https://hf-mirror.com/datasets/royokong/fashioniq_val_imgs/resolve/main/"
       "data/val-00000-of-00001.parquet?download=true")
OUT = os.path.join(OUT_DIR, "val-00000-of-00001.parquet")
SEG = 4 * 1024 * 1024          # 4 MiB per segment
THREADS = 12
HDR = {"User-Agent": "Mozilla/5.0"}
TRIES = 4


def _open(req):
    return urllib.request.urlopen(req, timeout=40)


def content_length():
    req = urllib.request.Request(URL, method="HEAD", headers=HDR)
    try:
        r = _open(req)
        n = int(r.headers.get("Content-Length", 0))
        r.close()
        return n
    except Exception as e:  # some hosts reject HEAD -> do ranged GET of 0 bytes
        print("HEAD failed:", repr(e)[:120], flush=True)
        req = urllib.request.Request(URL, headers=HDR)
        r = _open(req)
        n = int(r.headers.get("Content-Length", 0))
        r.close()
        return n


def seg_range(i, seg_len, total):
    a = i * seg_len
    b = min(total, (i + 1) * seg_len) - 1
    return a, b


def fetch(i, seg_len, total):
    part = OUT + f".part{i:04d}"
    a, b = seg_range(i, seg_len, total)
    need = b - a + 1
    try:
        if os.path.exists(part) and os.path.getsize(part) == need:
            return (i, need, True)
    except OSError:
        pass
    for t in range(TRIES):
        try:
            req = urllib.request.Request(URL, headers=dict(HDR, Range=f"bytes={a}-{b}"))
            with _open(req) as r:
                data = r.read()
            if len(data) != need:
                raise IOError(f"short {len(data)}!={need}")
            tmp = part + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, part)
            return (i, need, False)
        except Exception as e:
            last = e
            time.sleep(1.5 * (t + 1))
    raise RuntimeError(f"seg {i} failed after {TRIES} tries: {last!r}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = content_length()
    if total <= 0:
        print("cannot determine content length; abort", flush=True)
        sys.exit(1)
    nseg = (total + SEG - 1) // SEG
    print(f"total={total/1e6:.1f}MB  segs={nseg} seg={SEG//1048576}MB threads={THREADS}", flush=True)
    t0 = time.time()
    done = [0]
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(fetch, i, SEG, total) for i in range(nseg)]
        for i, fut in enumerate(futs):
            try:
                _, got, reused = fut.result()
                done[0] += got
                rate = done[0] / 1e6 / max(time.time() - t0, 1e-9)
                eta = (total - done[0]) / 1e6 / max(rate, 1e-9) / 60
                print(f"[{i+1}/{nseg}] got={got/1e6:.2f}MB total={done[0]/1e6:.1f}MB "
                      f"rate={rate:.2f}MB/s eta={eta:.0f}m reused={reused}", flush=True)
            except Exception as e:
                print("FAIL:", repr(e)[:200], flush=True)
                sys.exit(1)
    # concatenate in order
    with open(OUT + ".tmp", "wb") as out:
        for i in range(nseg):
            part = OUT + f".part{i:04d}"
            with open(part, "rb") as f:
                while True:
                    buf = f.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
    os.replace(OUT + ".tmp", OUT)
    size = os.path.getsize(OUT)
    print(f"DONE  final={OUT} size={size/1e6:.1f}MB wall={(time.time()-t0)/60:.1f}m", flush=True)
    if size != total:
        print("!!! size mismatch", flush=True)
        sys.exit(2)
    for i in range(nseg):  # cleanup parts
        try:
            os.remove(OUT + f".part{i:04d}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
