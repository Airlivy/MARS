# FashionIQ 小 Demo（纯 CLIP 组合检索交互界面）

为多阶段框架的 **Stage-1 检索** 提供一个直观的交互演示：点选一张**参考图** + 输入中文修改指令
（如「把蓝色改成红色」），系统在 **真实 FashionIQ 服装图库**（15536 张）中检索出「修改后最接近」的候选图。

引擎刻意保持**轻量、秒级**：纯 CLIP（laion ViT-B/32），**不走 Ray / vLLM / LLM / 重排序**。

## 数据

图库为**真实 FashionIQ val 服装商品图**。仓库不包含数据，需用 `tools/` 下的三个脚本自行生成，
默认落到 `data/FashionIQ_real/`（可用环境变量 `FIQ_REAL_DIR` 改到别处）：

```bash
python tools/fiq_download.py     # ① 从 hf-mirror 下载 val parquet（207MB，15536 行，支持断点续传）
python tools/fiq_decode.py       # ② 解出 15536 张 images/<id>.jpg（约 228MB）
python tools/fiq_build_db.py     # ③ 用 CLIP 编码全库 → clip_db 下的两个 npy（本机约 2 分钟）
```

生成的目录结构：

```
data/FashionIQ_real/
├── val-00000-of-00001.parquet   # ① 的产物
├── meta.jsonl                   # ② 的产物，{name,id,category}，category ∈ dress/shirt/toptee
├── images/                      # ② 的产物，15536 张 .jpg
└── clip_db/CLIP-ViT-B-32/       # ③ 的产物
    ├── image_name_list.npy      #   图名（与 server 加载顺序一致）
    └── image_embedding.npy      #   (15536,512) float32
```

- ② 对重复的图片 id 用 `<id>__<k>` 后缀去重，保证图名全局唯一。
- ③ 需要 torch / transformers，建议有 GPU；CLIP 权重走环境变量 `CLIP_PATH`，默认从 HuggingFace 拉取。
- 界面网格只展示部分（每批 240 张、可「加载更多」），**检索始终在全库 15536 张进行**。

## 目录

```
demo_fiq/
├── server.py          # FastAPI 后端：图库/图/检索 API + 中文指令→英文查询
├── requirements.txt   # demo 依赖（版本取自 src/Core/requirements.txt）
├── static/index.html  # 中文前端（自包含、无外部依赖）
├── find_cases.py      # 离线挑选演示案例的脚本（跑一次即可，运行期不依赖）
├── assets/            # 演示案例产物（由 find_cases.py 生成）
│   ├── cases.json          # 前端「推荐演示案例」按钮的数据（当前 2 个精选案例）
│   ├── cases_contact.png   # 全量候选接触板
│   ├── cases_final.png     # 2 个案例的实测拼图（见 演示案例.md）
│   ├── candidates.json     # 候选清单
│   └── case_results.json   # 案例实测结果
├── README.md
└── 演示案例.md         # 案例演示说明
```

## 运行

```bash
pip install -r src/demo_fiq/requirements.txt
cd src/demo_fiq                                # 在仓库根目录下执行
python -m uvicorn server:app --host 0.0.0.0 --port 8899
```

浏览器打开 **http://127.0.0.1:8899**。

CLIP 模型：默认用 HuggingFace 模型 ID `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`；
如已下载到本地，可用环境变量 `CLIP_PATH` 指向该目录以避免重复下载。
有 GPU 时自动用 cuda，否则回退 CPU（见 `server.py` 里 `_device` 的选择）。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/api/images?batch=N` | 分批发图库（每批 240 张，name/url，含 total/has_more） |
| GET | `/img/{name}` | 单张图（jpg/png，宽容扩展名） |
| POST | `/api/search` | 融合检索，body：`{name, inst_cn, top_k?, alpha?}` |
| GET | `/api/cases` | 推荐案例（读 `assets/cases.json`） |

## 检索原理

查询向量 = `normalize( ref_vec + α × text_vec )`，再与全库 15536 个向量点积排序（**排除参考图自身**）：

- `ref_vec`：直接取 `clip_db` 里该参考图已算好的 CLIP 图像向量，**无需实时编码**；
- `text_vec`：把中文指令用小词典翻译成英文（颜色/服装名词/改动词），再实时编码 1 条 CLIP 文本；
- `α`：参考图↔文本融合权重，前端滑杆可调（默认 0.7，推荐案例各有实测值）。

中→英翻译覆盖常见颜色、服装名词（衣服/连衣裙/衬衫/T恤/外套/裤子…）、「把 X 改成 Y / 变成 /
去掉 X」等句式；识别到颜色改色就翻译（并标记已识别），识别不到就退回原文本查询，不影响演示。

## 重新挑选演示案例（可选）

```bash
# 路径按脚本自身位置推导，在仓库根或 src/demo_fiq 下执行都可以
python src/demo_fiq/find_cases.py               # 全量判色+扫描 → assets/cases_contact.png + assets/candidates.json
python src/demo_fiq/find_cases.py --finalize    # 按 assets/cases.json 实测 → assets/cases_final.png + assets/case_results.json
```

真实商品照片常有背景/肤色/多色：判色只取**中央 70% 区域**的强饱和像素，且要求**单一主色占主导**
才当作候选参考图；案例是否“改对了色”同时用**像素主色**与 **CLIP 颜色+款式标签**双验证，
而非只看相似度自圆其说。
