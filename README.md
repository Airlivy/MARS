# MARS · 组合图像检索系统

**组合图像检索（Composed Image Retrieval, CIR）**：给定一张参考图 + 一句相对修改描述
（如「把蓝色改成红色」），在图像库中检索出「保留参考图主体、又应用了修改」的目标图像。

本仓库包含两部分：一个**多阶段 CIR 框架**，以及一个**纯 CLIP 的交互式检索 Demo**。

---

## 能跑什么

先说清楚，免得白下几十 GB：

| 部分 | 状态 | 前提 |
|---|---|---|
| **交互式检索 Demo** | ✅ 克隆后可跑 | 一张 GPU（没有也行，自动回退 CPU）、约 470MB 磁盘、可访问 HuggingFace |
| **多阶段 CIR 框架** | ⚠️ 需自备模型与数据集 | 4 个 8B 级模型（可指向同一份，约 16GB）、完整数据集、多卡，详见 [`src/README.md`](src/README.md) |

Demo 是这份仓库里**唯一能开箱即用**的部分，下面直接给步骤。
多阶段框架需要的外部资源本仓库不提供，用法见 [`src/README.md`](src/README.md) 第五节。

---

## 快速开始：交互式检索 Demo

在 15536 张**真实 FashionIQ 服装商品图**上做组合检索：点选一张参考图 + 输入中文修改指令
（如「把紫色改成蓝色」），秒级返回候选图。引擎刻意保持轻量——**纯 CLIP（laion ViT-B/32），
不走 Ray / vLLM / LLM / 重排序**，参考图向量全部离线预计算。

### 1. 装依赖

```bash
pip install -r src/demo_fiq/requirements.txt
```

PyTorch 的 wheel 与 CUDA 版本绑定，建议按自己的环境单独装，例如 CUDA 11.8：

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 准备数据（三步）

仓库不含数据，用 `tools/` 下的脚本生成。默认落到 `data/FashionIQ_real/`：

```bash
python tools/fiq_download.py     # ① 下载 val parquet         207MB，支持断点续传
python tools/fiq_decode.py       # ② 解出图片 + meta.jsonl    15536 张，约 228MB
python tools/fiq_build_db.py     # ③ 建 CLIP 图像向量库       本机约 2 分钟
```

第 ③ 步需要 GPU 才有速度；会从 HuggingFace 拉取 CLIP 权重，下载慢的话可先
`export HF_ENDPOINT=https://hf-mirror.com`，或用 `CLIP_PATH` 指向本地已下好的目录。
三步都完成后，`val-00000-of-00001.parquet` 就可以删了。

### 3. 启动

```bash
cd src/demo_fiq
python -m uvicorn server:app --host 0.0.0.0 --port 8899
```

浏览器打开 **http://127.0.0.1:8899** 。

### 4. 界面

- 网格里每批展示 240 张，可「加载更多」——但**检索始终在全库 15536 张上进行**
- 点选参考图 → 输入中文指令 → 调 `α` 滑杆（参考图与文本的融合权重）→ 检索
- 顶部有「推荐演示案例」按钮，数据来自 [`src/demo_fiq/assets/cases.json`](src/demo_fiq/assets/cases.json)

Demo 的原理、API 与案例挑选脚本见 [`src/demo_fiq/README.md`](src/demo_fiq/README.md)。

---

## 多阶段 CIR 框架

| 阶段 | 作用 |
|---|---|
| Stage-1 | LLM 对「参考图 + 修改描述」推理改写，经 CLIP 向量检索 + 多路融合（RRF）得到候选 |
| Stage-2 | MLLM 对候选做问答 / 文本蕴含式验证重排 |
| Stage-3 | 输出指标或用评估模型打点 |

入口是 `src/Core/main.py`，路径按自身文件位置推导，**在哪个目录执行都可以**：

```bash
python src/Core/main.py --dataset FashionIQ-dress --split valid --run_name fiq-dress \
  --llm_path OpenGVLab/InternVL3-8B --mllm_path OpenGVLab/InternVL3-8B \
  --img_cap_model_path OpenGVLab/InternVL3-8B --eval_mllm_path OpenGVLab/InternVL3-8B \
  --clip_path laion/CLIP-ViT-L-14-laion2B-s32B-b82K
```

四个模型参数可以全部指向同一份权重，框架用 Ray 按阶段调度，不会同时占满显存。

**当前可运行的数据集是 5 个**：`CIRCO`、`CIRR`、`FashionIQ-dress`、`FashionIQ-shirt`、
`FashionIQ-toptee`。`MSCOCO` / `Flickr30K` / `VisDial` 的 prompt 模板尚未补齐，
启动时会直接报错退出（报错信息会列出缺哪些文件）。

完整的安装、模型配置、数据集准备、参数手册与 FAQ，全部在 **[`src/README.md`](src/README.md)**。

---

## 目录结构

```
.
├── README.md
├── src/
│   ├── README.md                # ★ 框架完整文档：安装 / 模型 / 数据 / 参数 / 评估 / FAQ
│   ├── Core/                    # 多阶段 CIR 框架
│   │   ├── main.py              # 入口
│   │   ├── requirements.txt     # 依赖清单（CUDA 11.8 全量 freeze）
│   │   ├── prompts/             # 各数据集 prompt 模板（5 个数据集）
│   │   ├── scripts/             # 各数据集预置启动脚本
│   │   └── utils/               # 数据加载与 Ray 推理封装
│   └── demo_fiq/                # 交互式检索 Demo
│       ├── server.py            # FastAPI 后端
│       ├── requirements.txt     # demo 依赖（比 Core 小得多）
│       ├── static/index.html    # 中文前端（自包含）
│       ├── find_cases.py        # 离线挑选演示案例
│       ├── assets/              # 演示案例产物
│       └── 演示案例.md
├── tools/
│   ├── fiq_download.py          # Demo 数据：下载 parquet
│   ├── fiq_decode.py            # Demo 数据：解出图片
│   ├── fiq_build_db.py          # Demo 数据：建 CLIP 向量库
│   └── make_smoke_data.py       # 生成合成占位图，用于框架 s1→s3 联调
└── data/                        # 数据集（不进版本库，自行准备）
```

## 数据与模型

本仓库**不含**任何数据集与模型权重，两者都需要自行准备：

- **Demo 用的 FashionIQ 图库**：由 `tools/fiq_*.py` 自动下载与生成，见上文第二步
- **框架用的数据集**：按 [`src/README.md`](src/README.md) 第三节准备，放到仓库根的 `data/` 下
- **模型权重**：框架需要 4 个 LLM/MLLM 路径与 1 个 CLIP 路径，均可直接传 HuggingFace 模型 ID 自动下载

## 外部基线与参考文献

框架实现中参考了 TIRG、CLIP4Cir、Pic2Word 等开源工作，仅作为基线对比与参考文献索引，
非本仓库组成部分。上游第三方代码副本不在此仓库分发。
