## 🛠️ 一、环境安装

> **推荐环境**：Linux (Ubuntu 20.04+) + CUDA 11.8 + 单机多卡（≥ 2 × 24GB VRAM）。
> 单卡可运行，但多卡并行可显著缩短推理时间。

### 1.1 前置检查

```bash
# 确认 CUDA 版本（需 ≥ 11.8）
nvcc --version

# 确认 GPU 可见
nvidia-smi

# 确认 conda 已安装
conda --version
```

### 1.2 创建 conda 环境

```bash
conda create -n project_env python=3.10.14 -y
conda activate project_env
```

### 1.3 安装 PyTorch 与 torchvision

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 1.4 安装性能加速组件

```bash
# xFormers（必须与 CUDA 11.8 对齐）
pip install xformers==0.0.27.post2 \
  --index-url https://download.pytorch.org/whl/cu118

# vLLM（预编译 wheel，避免源码编译）
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.4/vllm-0.5.4+cu118-cp310-cp310-manylinux1_x86_64.whl
```

### 1.5 安装项目依赖

依赖清单在 `src/Core/requirements.txt`，以下命令在**仓库根目录**执行：

```bash
pip install -r src/Core/requirements.txt
```

> ⚠️ 请**按 1.3 → 1.4 → 1.5 的顺序**安装。清单里的 `torch==2.4.0+cu118`、
> `torchvision==0.19.0+cu118`、`xformers==0.0.27.post2+cu118` 带 CUDA 本地版本号，
> 这些 wheel 只存在于 PyTorch 专属 index、PyPI 上没有，单独跑本步骤会解析失败；
> 先按 1.3 / 1.4 装好，本步骤就只是补齐其余依赖。

### 1.6 安装注意事项

| 问题 | 说明 |
|------|------|
| 🔒 ABI 版本冲突 | 严格保持 `torch`、`xformers`、`vLLM` 的 CUDA 构建版本一致；混用不同 CUDA 版本（如 torch cu118 + vLLM cu120）会导致 `RuntimeError: CUDA error: invalid device function`。 |
| 💾 HuggingFace 缓存 | 默认路径为 `~/.cache/huggingface`。若磁盘紧张，请设置环境变量重定向：<br>`export HF_HOME=/your/large/disk/huggingface` |
| 🌐 国内镜像加速 | 下载 HuggingFace 模型时建议配置镜像：<br>`export HF_ENDPOINT=https://hf-mirror.com` |
| 🐛 常见报错 | 若遇到 `libcuda.so.1: cannot open shared object file`，请确认 `nvidia-smi` 正常输出且已安装对应版本的 NVIDIA 驱动。 |

---

## 🧠 二、模型路径配置

框架支持两种模型加载方式：

1. **HuggingFace Hub 在线加载**（推荐，首次自动下载）：直接传入模型 ID，如 `OpenGVLab/InternVL3-8B`
2. **本地绝对路径加载**：传入已下载好的模型目录，如 `/mnt/models/InternVL3-8B`

### 2.1 模型参数说明

以下参数在 `main.py` 中通过命令行传入，或在 `scripts/run_*.sh` 中批量修改。

| 参数 | 作用 | 推荐模型 | 显存占用（单实例） |
|---|---|---|---|
| `--llm_path` | 第一阶段文本推理器（Reasoner） | `OpenGVLab/InternVL3-8B` | ~16 GB |
| `--mllm_path` | 第二阶段验证用 MLLM（Verifier） | `OpenGVLab/InternVL3-8B` | ~16 GB |
| `--img_cap_model_path` | Captioner MLLM（构建 `image_db/` 时生成图像描述） | `OpenGVLab/InternVL3-8B` | ~16 GB |
| `--eval_mllm_path` | 第三阶段评估 MLLM（Evaluator） | `OpenGVLab/InternVL3-8B` | ~16 GB |
| `--clip_path` | CLIP 视觉编码器（用于图像特征提取与检索） | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | ~2 GB |

> 💡 **节省显存/磁盘技巧**：5 个模型可全部指向同一份 `InternVL3-8B` 权重。框架内部通过 Ray 调度，同一时刻不会全部加载到显存，而是在不同阶段按需加载。

### 2.2 本地模型路径优先级

框架内部通过 `_resolve_model_path` 函数解析路径，优先级如下：

1. 命令行显式传入的路径
2. 环境变量默认值（如已设置）
3. 回退到 HuggingFace 标准 ID：`OpenGVLab/InternVL3-8B`

### 2.3 模型下载命令（可选）

```bash
# 使用 huggingface-cli 提前下载（适合离线环境）
pip install huggingface-hub
huggingface-cli download OpenGVLab/InternVL3-8B --local-dir /path/to/local/InternVL3-8B
huggingface-cli download laion/CLIP-ViT-L-14-laion2B-s32B-b82K --local-dir /path/to/local/CLIP-ViT-L-14
```

---

## 📂 三、数据集准备

### 3.1 数据根目录结构

在任意目录下创建数据文件夹（默认名为 `data/`，可通过 `--dataset_path` 修改）：

```
data/
├── CIRCO/
├── CIRR/
├── FashionIQ/
├── Flickr30K/
├── MSCOCO/
└── VisDial/
```

### 3.2 各数据集详细准备指南

| 数据集 | 下载来源 | 目录结构要求 | 备注 |
|---|---|---|---|
| **CIRCO** | [官方 GitHub](https://github.com/teranado/CIRCO) | `captions/{val.json, test.json}` + `unlabeled2017/*.jpg` | 图像复用 MSCOCO 2017 unlabeled |
| **CIRR** | [官方下载](https://github.com/ABaldrati/CLIP4CIR) | `captions/`、`captions_ext/`、`image_splits/`、`dev/`、`test1/` | 支持子集评估：`--subset` |
| **FashionIQ** | [官方下载](https://github.com/XiaoxiaoGuo/fashion-iq) | `images/` + `image_splits/split.{dress,shirt,toptee}.val.json` + `captions/cap.{dress,shirt,toptee}.val.json` | 分 dress / shirt / toptee 三个子任务 |
| **Flickr30K** | [Kaggle / 官方](http://shannon.cs.illinois.edu/DenotationGraph/) | 按官方目录结构放置 | ⚠️ **prompt 模板缺失，暂不可运行** |
| **MSCOCO** | [官方下载](https://cocodataset.org/) | 按官方目录结构放置 | ⚠️ **prompt 模板缺失，暂不可运行** |
| **VisDial** | [官方下载](https://visualdialog.org/) | 按官方目录结构放置 | ⚠️ **prompt 模板缺失，暂不可运行** |

> ⚠️ **实际可运行的数据集为 5 个**：`CIRCO`、`CIRR`、`FashionIQ-dress/shirt/toptee`。
> `src/Core/prompts/` 下只有这 5 个数据集的模板，而 `main.py` 启动时会强制读取
> `prompts/{dataset}/` 下的 4 个必需文件（`prompt1_stage1_reasoner.txt`、
> `prompt2_stage2_reasoner.txt`、`caption.txt`、`prompt3_stage3_evaluator.txt`），
> 缺失会直接报错退出。上表后三个数据集的下载方式保留作参考，但**补齐模板前无法运行**。

### 3.3 验证数据集是否就绪

以下路径相对**仓库根目录**（即 `data/` 所在处）：

```bash
# 检查 FashionIQ 图像数量（ dress 约 30425 张）
ls data/FashionIQ/images/ | wc -l

# 检查 CIRR 标注文件
head -5 data/CIRR/captions/cap.rc2.test1.json
```

---

## ⚙️ 四、完整参数手册

### 4.1 模型路径参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--llm_path` | `str` | `OpenGVLab/InternVL3-8B` | 第一阶段文本推理器模型路径 |
| `--mllm_path` | `str` | `OpenGVLab/InternVL3-8B` | 第二阶段验证 MLLM 路径 |
| `--img_cap_model_path` | `str` | `OpenGVLab/InternVL3-8B` | Captioner 模型路径（生成 image_db 描述） |
| `--eval_mllm_path` | `str` | `OpenGVLab/InternVL3-8B` | 第三阶段评估 MLLM 路径 |
| `--clip_path` | `str` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP 视觉编码器路径 |
| `--dataset_path` | `str` | `data/` | 数据集根目录 |

### 4.2 第一阶段（Stage-1）参数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--stage1_retrieve_texts` | `str` | `v1,v2,aug` | `v1`, `v2`, `aug` 及组合 | 并检索的文本源，逗号分隔。`v1` = 标准结构化描述，`v2` = cir-v2 格式，`aug` = 扩增描述 |
| `--stage1_fusion` | `str` | `rrf` | `none`, `rrf`, `interleave` | 多路检索结果融合策略：`rrf` = 倒数秩融合，`interleave` = 交错排序，`none` = 不融合 |
| `--rrf_k` | `int` | `60` | — | RRF 融合常数 k，越大对高排名的惩罚越小 |
| `--stage1_schema` | `str` | `both` | `v1`, `v2`, `both` | Stage-1 输出格式：`v1` = step1/step2 结构，`v2` = cir-v2 细节，`both` = 同时输出两种 |
| `--stage1_rewrite_caption` | `bool` | `True` | — | Stage-1 结束后是否用 cir-v2 细节重写最终目标描述 |

### 4.3 第二阶段（Stage-2）参数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--stage2_source` | `str` | `aug` | `auto`, `v1`, `v2`, `aug`, `fused` | 喂入 Stage-2/3 的检索列表来源。`auto` = 优先 fused，否则取第一个变体 |
| `--s2_num_questions` | `int` | `None` | — | 限制 Stage-2 生成原子问题的数量。`None` 表示不限制 |
| `--s2_mode` | `str` | `both` | `qa`, `caption`, `both` | Stage-2 验证模式：`qa` = MLLM 看图问答，`caption` = 文本蕴含判断，`both` = 同时运行两种并融合 |

### 4.4 通用控制参数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--tau` | `float` | `0.15` | — | 温度参数，控制检索/重排序的锐度 |
| `--top_k` | `int` | `100` | — | 检索返回的候选图像数量 |
| `--alpha` | `int` | `100` | — | 重排序权重系数 |
| `--stages` | `list` | `s1 s2` | `s1`, `s2`, `s3` | 要执行的阶段，可组合如 `s1 s2 s3` |
| `--dataset` | `str` | — | `CIRR`, `CIRCO`, `FashionIQ-dress`, `FashionIQ-shirt`, `FashionIQ-toptee`（可用）；`MSCOCO`, `Flickr30K`, `VisDial`（⚠️ prompt 模板缺失，暂不可运行） | 目标数据集 |
| `--split` | `str` | `valid` | `valid`, `test` | 数据划分：FashionIQ / VisDial 用 `valid`，其余用 `test` |
| `--subset` | `bool` | `False` | — | 仅对 `CIRR` 有效，使用子集评估 |
| `--run_name` | `str` | `default` | — | 本次运行的标识名，用于生成输出目录 |
| `--force` | `bool` | `False` | — | 强制覆盖已有输出目录 |

### 4.5 断点续跑参数（高级）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--restore_states` | `str` | `None` | — | 从指定的 `states.pkl` 文件恢复运行状态 |
| `--stage_skip_num` | `int` | `0` | `0, 1, 2, 3, 4` | 恢复时跳过前 N 个 stage（与 `restore_states` 配合使用） |

> 💡 **断点续跑典型场景**：Stage-1 已跑完，想从 Stage-2 开始：<br>`--restore_states src/Core/runs/CIRR_test/CLIP-ViT-L-14-fiqq_20240115_120000/states.pkl --stage_skip_num 2`

---

## ▶️ 五、运行推理

> 📌 `prompts/`、`image_db/`、`runs/` 均锚定在 `src/Core/` 下（由 `main.py` 按自身文件位置推导），
> 因此**在哪个目录执行都可以**，输出始终落在 `src/Core/runs/`。
> `--dataset_path` 默认指向仓库根的 `data/`，数据集放在别处时才需要显式传入。

### 5.1 快速启动：FashionIQ-dress 验证集

```bash
conda activate project_env

# 在仓库根目录执行（也可以先 cd src/Core，再 python main.py，效果相同）
CUDA_VISIBLE_DEVICES=0,1 python src/Core/main.py \
  --dataset FashionIQ-dress \
  --split valid \
  --run_name fiq-dress \
  --s2_num_questions 1 \
  --llm_path OpenGVLab/InternVL3-8B \
  --mllm_path OpenGVLab/InternVL3-8B \
  --img_cap_model_path OpenGVLab/InternVL3-8B \
  --eval_mllm_path OpenGVLab/InternVL3-8B \
  --clip_path laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
  --dataset_path /path/to/data   # 可选；默认已指向 <仓库根>/data
```

### 5.2 使用预置脚本（推荐）

`src/Core/scripts/` 已预置各数据集的启动脚本，只需修改模型路径和数据集路径即可：

```bash
# 编辑脚本，替换占位符为实际路径
vim src/Core/scripts/run_FashionIQ.sh

# 执行（脚本内为 `python main.py`，需在 src/Core 下运行）
cd src/Core && bash scripts/run_FashionIQ.sh
```

| 脚本 | 说明 |
|---|---|
| `run_FashionIQ.sh` | 同时运行 dress、shirt、toptee 三个子任务 |
| `run_CIRR.sh` | CIRR 测试集；加 `--subset` 跑子集 |
| `run_CIRCO.sh` | CIRCO 测试集 |
| `run_Flickr30K.sh` | Flickr30K 验证集（⚠️ prompt 模板缺失，暂不可运行） |
| `run_MSCOCO.sh` | MSCOCO 验证集（⚠️ prompt 模板缺失，暂不可运行） |
| `run_VisDial.sh` | VisDial 多轮对话验证集（⚠️ prompt 模板缺失，暂不可运行） |

### 5.3 运行说明

| 问题 | 说明 |
|---|---|
| 🎛️ GPU 调度 | 框架通过 Ray 自动调度所有可见 GPU。如需限制，显式设置 `CUDA_VISIBLE_DEVICES=0,1`。 |
| 🗂️ image_db 构建 | 每个 `数据集 + CLIP 版本` 首次运行会自动构建 `image_db/{dataset}/{clip_version}/`（包含 captions 和 embeddings）。构建后复用，无需重复生成。 |
| 🧪 CIRR 子集 | 在命令末尾追加 `--subset` 即可评估子集性能。 |
| 🧩 多轮对话 | VisDial 场景会自动执行多轮推理，每轮结果保存在 `states.pkl` 中。 |

---

## 📊 六、输出结果与评估

### 6.1 输出目录结构

```
runs/
└── {dataset}_{split}/
    └── {clip_version}-{run_name}_{timestamp}/
        ├── output.log          # 运行日志与指标
        ├── states.pkl          # 中间状态（可恢复运行）
        ├── prompt1_stage1_reasoner.txt
        ├── prompt2_stage2_reasoner.txt
        ├── caption.txt
        └── prompt3_stage3_evaluator.txt
```

### 6.2 评估方式

| 数据集 | 评估方式 |
|---|---|
| **FashionIQ / Flickr30K / MSCOCO / VisDial** | 直接读取 `output.log` 中的 `Hits@k` 指标。 |
| **CIRCO / CIRR** | 将 `output.log` 旁生成的 `{timestamp}_{dataset}_test_stage3_eval.json` 提交到对应官方评测服务器。 |

### 6.3 指标解读

- `Hits@k`：目标图像在检索结果中排名前 k 的样本比例。k 取值通常为 1, 2, 3, 4, 5, 10, 20, 25, 50。
- `Recall@k` / `R-Precision`：部分数据集额外报告的指标，详见 `output.log`。

---

## 🗂️ 七、项目目录结构

```
MARS/
├── README.md
├── src/
│   ├── README.md                       # 本文件
│   ├── Core/                           # 多阶段 CIR 框架
│   │   ├── main.py                     # 主入口（需在 Core/ 目录下运行）
│   │   ├── requirements.txt            # 依赖清单
│   │   ├── prompts/                    # 各数据集的 prompt 模板
│   │   │   ├── FashionIQ-dress/
│   │   │   ├── CIRR/
│   │   │   └── ...
│   │   ├── scripts/                    # 预置启动脚本
│   │   │   ├── run_CIRCO.sh
│   │   │   ├── run_CIRR.sh
│   │   │   ├── run_FashionIQ.sh
│   │   │   ├── run_Flickr30K.sh
│   │   │   ├── run_MSCOCO.sh
│   │   │   └── run_VisDial.sh
│   │   ├── utils/                      # 工具函数与推理封装
│   │   │   ├── data.py
│   │   │   ├── function.py
│   │   │   └── inference_entrypoint_ray.py
│   │   ├── image_db/                   # 图像数据库（首次运行自动生成）
│   │   │   └── {dataset}/{clip_version}/
│   │   └── runs/                       # 运行输出（自动创建）
│   └── demo_fiq/                       # FashionIQ 交互式检索 Demo
│       ├── README.md
│       ├── server.py                   # FastAPI 后端
│       ├── find_cases.py               # 离线挑选演示案例
│       ├── static/index.html           # 前端页面
│       └── assets/                     # 演示案例产物
├── tools/
│   ├── fiq_download.py                 # 下载 FashionIQ val parquet（Demo 用）
│   ├── fiq_decode.py                   # 解出图片 + meta.jsonl（Demo 用）
│   ├── fiq_build_db.py                 # 建 CLIP 向量库 clip_db（Demo 用）
│   └── make_smoke_data.py              # 生成合成占位图，用于 s1→s3 联调
└── data/                               # 数据集（不进版本库，自行准备）
    └── {CIRCO,CIRR,FashionIQ,...}/
```

> 📌 `image_db/` 与 `runs/` 位于 `src/Core/` 下：`main.py` 按**自身文件位置**推导这些路径
> （不是按当前工作目录），所以在哪个目录执行都落在同一处。

---

## ❓ 八、常见问题排查（FAQ）

### Q1: 首次运行卡住不动？
- 大概率是在下载 HuggingFace 模型。检查网络或配置 `HF_ENDPOINT=https://hf-mirror.com`。

### Q2: `CUDA out of memory`？
- 减少 `--top_k`（如改为 50）
- 限制 `--s2_num_questions`（如 1）
- 确认 `--stage1_retrieve_texts` 不要设置过多并行源
- 确保多卡环境下 Ray 能看到所有 GPU

### Q3: `image_db` 构建失败？
- 检查数据集路径是否正确，图像文件是否存在
- 检查 `prompts/{dataset}/caption.txt` 是否存在

### Q4: 如何只跑 Stage-1 做快速验证？
```bash
python src/Core/main.py --stages s1 --dataset FashionIQ-dress ...
```

### Q5: 如何恢复上次运行从 Stage-2 继续？
```bash
python src/Core/main.py \
  --restore_states src/Core/runs/.../states.pkl \
  --stage_skip_num 2 \
  ...
```

### Q6: 更换 CLIP 版本后需要重新构建 image_db 吗？
- 是的。`image_db` 按 `{dataset}/{clip_version}/` 隔离存储，更换 CLIP 会自动创建新目录并重新构建。

### Q7: 运行日志中出现大量 JSON decode fail？
- 通常不影响最终结果，框架会自动跳过并记录。若比例过高，可检查 LLM 输出是否被截断（调整 `--tau` 或模型温度参数）。

---

## 🔗 附录：外部基线与相关项目

本框架在实现中参考了以下开源工作（仅作为基线对比，非本项目组成部分）：

- CLIP4Cir / Pic2Word / CIRPLANT / MAAF / COMMA2 / SEARLE / SEARLE-XL / CiReR / R.Q. / FICOR / FashionVLP / TransR / SEARLE-v2 / MAAFi / CLIP4CIR / AMCDN / CLIP4CIR-RN / FSAR / GAL-VI / L-Y / MA-FE / [CIR] / SEARLE-v3 / AM-CIR / SEARLE / [COR] / RF / MCR / F-Minus / X2CIR / W-V / AR-ViT / MA-GAT / TIRG / B-S / [CIR] / SSM / MAAFi / (V) / GAL-Vi / S-MCR / ATTIRE / M-X / L-Y / QR / MA-FE

> ⚠️ 以上基线数据保留原始论文引用格式，仅用于实验对比与参考文献索引。