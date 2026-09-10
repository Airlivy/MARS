# -*- coding: utf-8 -*-
"""在真实服装图库上挑 1~2 个效果稳的演示案例（配合 server.search 同款融合数学）。

真实商品照片常有背景/肤色/多色：判色只取「中央 70% 区域」的强饱和像素，
且只把「单一主色占主导」的图当作候选参考图，避免被背景或花纹误导。

用法（需 torch/transformers，建议 GPU）：
  python find_cases.py               # 全量扫描 -> assets/cases_contact.png + assets/candidates.json
  python find_cases.py --finalize    # 按 assets/cases.json 实测 -> assets/cases_final.png + assets/case_results.json

输出里每张图的「主色」是客观像素判色（HSV 中央区）；另有 CLIP 文本近邻标签（颜色+款式）供写说明。
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402  复用 IMAGE_NAMES / IMAGE_EMB / 路径

# ---------------- 色板（判色共用；名字 -> 中文 + 色相中心°） ----------------
PALETTE = {
    "red":    ("红色", 0),
    "orange": ("橙色", 30),
    "yellow": ("黄色", 60),
    "green":  ("绿色", 120),
    "cyan":   ("青色", 180),
    "blue":   ("蓝色", 240),
    "purple": ("紫色", 275),
    "pink":   ("粉色", 330),
}
HUE_CENTERS = {en: d for en, (_cn, d) in PALETTE.items()}
CN2EN = {en: cn for en, (cn, _) in PALETTE.items()}
TO_COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]  # 目标不含 pink(易与红混)
ALPHAS = [0.6, 0.9, 1.2, 1.6]
TOP_K = 6
CROP = 0.70          # 取中央 70% 判色
SAT_MIN, VAL_MIN = 80, 55
HUE_TOL = 26         # ±26° 归属
MIN_DOM_FRAC = 0.06  # 主色至少占整图这样比例
MIN_DOM_SHARE = 0.45  # 主色至少占「饱和像素」这样比例（排除多色/花纹）
SELF_DIR = server.SELF_DIR
ASSETS_DIR = server.ASSETS_DIR
os.makedirs(ASSETS_DIR, exist_ok=True)
CONTACT_PNG = os.path.join(ASSETS_DIR, "cases_contact.png")


def load_meta():
    """name -> category（FashionIQ_real/meta.jsonl；合成库没有则空）。"""
    meta_path = os.path.join(server.ROOT, "data", "FashionIQ_real", "meta.jsonl")
    cat = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    cat[d["name"]] = d["category"]
    return cat


CAT = load_meta()


def _center_hsv(name, size=192):
    p = server.img_path(name)
    im = Image.open(p).convert("RGB")
    w, h = im.size
    cw, ch = int(w * CROP), int(h * CROP)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.LANCZOS)
    return np.asarray(im.convert("HSV"))


def detect(name):
    """返回 dict：主色名/主色像素/饱和像素/图内占比。无色相主色则 dominant=None。"""
    hsv = _center_hsv(name)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    px = h.shape[0] * h.shape[1]
    mask = (s > SAT_MIN) & (v > VAL_MIN)
    n_sat = int(mask.sum())
    deg = h[mask].astype(np.float32) / 255.0 * 360.0
    counts = {}
    for en in PALETTE:
        c = HUE_CENTERS[en]
        d = np.abs(((deg - c + 180.0) % 360.0) - 180.0)
        counts[en] = int((d <= HUE_TOL).sum())
    if n_sat == 0:
        return {"dominant": None, "dom_px": 0, "sat": 0, "frac": 0.0}
    dom = max(counts, key=counts.get)
    dom_px = counts[dom]
    frac = dom_px / px
    share = dom_px / n_sat
    if frac < MIN_DOM_FRAC or share < MIN_DOM_SHARE:
        dom = None
    return {"dominant": dom, "dom_px": int(dom_px), "sat": n_sat,
            "frac": round(float(frac), 4), "share": round(float(share), 3)}


def is_clean_solid(d):
    return d["dominant"] is not None


# ---------------- 扫描 ----------------
def main():
    t0 = time.time()
    n = len(server.IMAGE_NAMES)
    print(f"[1/3] 判色 {n} 张 ...", flush=True)
    det = {}
    solid = []
    for i, name in enumerate(server.IMAGE_NAMES):
        d = detect(name)
        det[name] = d
        if is_clean_solid(d):
            solid.append(name)
        if (i + 1) % 4000 == 0:
            print(f"   {i+1}/{n} solid={len(solid)}", flush=True)
    print(f"   done {time.time()-t0:.0f}s, solid={len(solid)}", flush=True)

    print("[2/3] 编码 to 色文本向量 ...", flush=True)
    text_vec = {}
    for en in TO_COLORS:
        text_vec[en] = server.encode_text(en)[0].astype(np.float32)

    lib = server.IMAGE_EMB
    lib = lib / np.maximum(np.linalg.norm(lib, axis=1, keepdims=True), 1e-9)
    idx = {name: i for i, name in enumerate(server.IMAGE_NAMES)}

    print("[3/3] 扫描 改色 × alpha ...", flush=True)
    cands = []
    for name in solid:
        from_en = det[name]["dominant"]
        ref_vec = lib[idx[name]]
        for to_en in TO_COLORS:
            if to_en == from_en:
                continue
            for alpha in ALPHAS:
                fused = ref_vec + alpha * text_vec[to_en]
                fused = fused / max(np.linalg.norm(fused), 1e-9)
                sim = lib @ fused
                order = np.argsort(-sim)
                order = order[order != idx[name]][:TOP_K]
                tops = [server.IMAGE_NAMES[j] for j in order]
                hits = [det[r]["dominant"] == to_en for r in tops]
                nh = sum(hits)
                metric = nh * 1000 + (500 if hits and hits[0] else 0) + float(sum(sim[j] for j in order))
                cands.append({
                    "ref": name, "cat": CAT.get(name, ""), "from": from_en, "to": to_en,
                    "alpha": alpha, "metric": round(metric, 1), "nh": nh, "first": bool(hits and hits[0]),
                    "tops": tops, "hitpat": "".join("1" if h else "0" for h in hits),
                    "sims": [round(float(sim[j]), 3) for j in order],
                    "tops_dom": [det[r]["dominant"] or "-" for r in tops],
                    "ref_frac": det[name]["frac"],
                })
    cands.sort(key=lambda c: (-c["metric"], -c["nh"]))

    print("\n=== 候选 top 12（ref 去重）===", flush=True)
    shown_ref = set()
    for c in cands:
        if c["ref"] in shown_ref:
            continue
        shown_ref.add(c["ref"])
        print(f"  ref={c['ref']}({c['cat']}/{CN2EN[c['from']]}) 把{CN2EN[c['from']]}改成{CN2EN[c['to']]} "
              f"a={c['alpha']} hits={c['hitpat']}({c['nh']}/{TOP_K})  frac={c['ref_frac']}")
        print(f"      tops: " + " ".join(f"{t}({d})" for t, d in zip(c["tops"], c["tops_dom"])))
        if len(shown_ref) >= 12:
            break

    # 接触板：8 个去重 ref 里挑指标最好的，每行 ref | 指令 | top-6
    seen, chosen = set(), []
    for c in cands:
        if c["ref"] in seen or len(chosen) >= 8:
            continue
        seen.add(c["ref"])
        chosen.append(c)

    thumb, gap, lh = 96, 5, 18
    rows = len(chosen)
    w = (1 + 1 + TOP_K) * (thumb + gap) + gap
    h = rows * (lh + thumb + gap) + gap
    sheet = Image.new("RGB", (w, h), (248, 249, 252))
    dr = ImageDraw.Draw(sheet)
    for rr, c in enumerate(chosen):
        cy = rr * (lh + thumb + gap) + gap
        dr.text((gap, cy), f"#{rr+1} {c['ref']}  {CN2EN[c['from']]}->{CN2EN[c['to']]} a={c['alpha']} "
                            f"hits={c['hitpat']}", fill=(50, 60, 80))
        for j, nm in enumerate([c["ref"]] + c["tops"]):
            im = Image.open(server.img_path(nm)).resize((thumb, thumb), Image.LANCZOS)
            x = j * (thumb + gap) + gap
            y0 = cy + lh
            sheet.paste(im, (x, y0))
            if j == 0:
                dr.rectangle([x, y0, x + thumb, y0 + thumb], outline=(0, 128, 90), width=3)
            else:
                dr.rectangle([x, y0, x + thumb, y0 + thumb], outline=(205, 210, 222), width=1)
    sheet.save(CONTACT_PNG)
    with open(os.path.join(ASSETS_DIR, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=1)
    print(f"\n接触板已存: {CONTACT_PNG}（每行 = 绿框参考图 + top-{TOP_K}）", flush=True)


# ---------------- CLIP 文本标签（颜色+款式，用于写说明） ----------------
GARMENTS = ["dress", "skirt", "t-shirt", "shirt", "blouse", "sweater", "hoodie",
            "jacket", "coat", "cardigan", "jeans", "pants", "shorts", "top"]
_text_templates = None


def _label_vecs():
    global _text_templates
    if _text_templates is not None:
        return _text_templates
    cols = [en for en in PALETTE]
    tmpl = [f"a {c} {g}" for c in cols for g in GARMENTS] + ["a photo of clothing"]
    vecs = np.stack([server.encode_text(t)[0] for t in tmpl])
    _text_templates = (tmpl, vecs)
    return _text_templates


def clip_label(name):
    """CLIP 文本近邻：返回最像的 '颜色 款式' 描述。"""
    tmpl, vecs = _label_vecs()
    v = server.IMAGE_EMB[server.NAME2IDX[name]]
    sims = vecs @ v
    best = int(np.argmax(sims))
    return tmpl[best][2:] if best < len(tmpl) - 1 else "clothing"


# ---------------- 按 cases.json 实测出拼图/数据 ----------------
def finalize():
    with open(os.path.join(ASSETS_DIR, "cases.json"), "r", encoding="utf-8") as f:
        cases = json.load(f)
    det_cache = {}

    def d(name):
        if name not in det_cache:
            det_cache[name] = detect(name)
        return det_cache[name]

    thumb, gap, lh, under = 128, 6, 30, 20   # under: 缩略图下方写 rank/score 的两行
    rows = len(cases)
    n_res = 8
    w = (1 + n_res) * (thumb + gap) + gap
    h = rows * (lh + thumb + under + gap) + gap
    sheet = Image.new("RGB", (w, h), (250, 251, 253))
    dr = ImageDraw.Draw(sheet)
    summary = []

    for rr, c in enumerate(cases):
        res = server.search(c["name"], c["inst"], top_k=n_res, alpha=c["alpha"])
        cy = rr * (lh + thumb + under + gap) + gap
        ref_d = d(c["name"])
        label = clip_label(c["name"])
        dr.text((gap, cy), f"Case{rr+1}  ref={c['name']} ({CAT.get(c['name'],'')})  "
                           f"pix={CN2EN[ref_d['dominant']] if ref_d['dominant'] else '?'}  "
                           f"clip≈{label}  query=\"{res['query_en']}\" a={res['alpha']}  {res['ms']}ms",
                fill=(25, 35, 55))
        im = Image.open(server.img_path(c["name"])).resize((thumb, thumb), Image.LANCZOS)
        x0, y0 = gap, cy + lh
        sheet.paste(im, (x0, y0))
        dr.rectangle([x0, y0, x0 + thumb, y0 + thumb], outline=(0, 128, 90), width=3)
        row = {"ref": c["name"], "cat": CAT.get(c["name"], ""), "inst": c["inst"],
               "query_en": res["query_en"], "alpha": res["alpha"],
               "ref_pix_dom": ref_d["dominant"], "ref_clip": label, "results": []}
        for j, nm in enumerate(res["names"]):
            im = Image.open(server.img_path(nm)).resize((thumb, thumb), Image.LANCZOS)
            x = (1 + j) * (thumb + gap) + gap
            sheet.paste(im, (x, y0))
            dd = d(nm)
            domj = CN2EN[dd["dominant"]] if dd["dominant"] else "-"
            dr.rectangle([x, y0, x + thumb, y0 + thumb], outline=(210, 215, 226), width=1)
            dr.text((x, y0 + thumb + 2), f"#{j+1} {nm}", fill=(60, 70, 90))
            dr.text((x, y0 + thumb + 13), f"{res['scores'][j]:.3f} {domj}", fill=(120, 132, 152))
            row["results"].append({"name": nm, "score": res["scores"][j],
                                   "pix_dom": dd["dominant"], "clip": clip_label(nm)})
        summary.append(row)

    out_png = os.path.join(ASSETS_DIR, "cases_final.png")
    sheet.save(out_png)
    print("拼图已存:", out_png)
    with open(os.path.join(ASSETS_DIR, "case_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for s in summary:
        doms = [r["pix_dom"] or "-" for r in s["results"]]
        clips = [r["clip"] for r in s["results"]]
        print(f"\nCase: ref={s['ref']}({s['cat']}) pix_dom={s['ref_pix_dom']} clip≈{s['ref_clip']} "
              f"query=\"{s['query_en']}\" a={s['alpha']}")
        print("  rank | name      | sim    | pix | clip")
        for i, r in enumerate(s["results"], 1):
            print(f"   {i}   | {r['name'][:18]:<18} | {r['score']:.3f} | {r['pix_dom'] or '-':<4} | {r['clip']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize()
    else:
        main()
