# KV Cache Offloading Research 环境与 GitHub 仓库完整教程

> 目标：在当前隔离式 HPC/Slurm/A100 环境上，搭建一个**可复现、可删除、可迁移、适合修改 vLLM 源码做 KV cache offloading research** 的工作流。  
> 当前重点不是“把 vLLM 装起来玩一下”，而是建立一个后续换机器、换组、继续做实验时仍然能复现的研究工程。

---

## 1. 当前环境事实

### 1.1 节点分工

当前集群有两类节点：

```text
admin 节点：
- 可以访问公网
- 用于下载 GitHub repo、Python 包、Hugging Face 模型、Docker/Apptainer 镜像
- 不跑正式 GPU 实验

comput10 计算节点：
- 不允许访问公网
- 管理员明确说明：计算节点特意不配网络，避免安全风险
- 用于 GPU 推理实验、KV cache offloading 实验
- 所有代码、模型、cache、实验输出应放在本地 NVMe
```

因此后续原则是：

```text
admin 下载 / 构建 / 打包
        ↓
rsync/scp 传到 comput10
        ↓
comput10 离线解包 / 离线运行
```

不要在 `comput10` 上执行：

```bash
yum install ...
pip install ...
uv pip install ...
git clone https://...
wget ...
curl ...
huggingface-cli download ...
```

---

### 1.2 comput10 的关键环境

已经确认：

```text
OS: CentOS Linux 7.9.2009
Kernel: 3.10.0
GPU: NVIDIA A100-PCIE-40GB
NVIDIA Driver: 535.54.03
nvidia-smi CUDA Version: 12.2
本地 NVMe: /ssdcache
```

`comput10` 没有 default route：

```text
ping 223.5.5.5
connect: Network is unreachable
```

这是安全策略，不要尝试给计算节点配公网。

---

### 1.3 本地盘选择

适合放实验数据的路径：

```text
/ssdcache
```

不要把模型、KV offload 文件、实验日志放到：

```text
/public     # NFS
/data_test  # ParaStor / 分布式文件系统
/home       # 不作为热路径
/tmp        # root 盘，不作为大量 offload 热路径
```

统一工作目录：

```bash
/ssdcache/zizhaowang/vllm_lab
```

推荐目录结构：

```text
/ssdcache/zizhaowang/vllm_lab/
├── envs/              # 离线 Python 环境
├── repos/             # vLLM 源码 / 你的研究代码
├── models/            # 模型权重
├── wheelhouse/        # 离线 wheel 包
├── images/            # Apptainer/Singularity 镜像
├── runs/              # 每次实验输出
├── logs/              # Slurm 日志
├── tmp/               # 临时文件
├── kv_cache_offload/  # KV cache disk offloading 目录
├── offline_pkgs/      # 从 admin 传来的 tar 包
├── hf_cache/          # Hugging Face cache
├── vllm_cache/        # vLLM cache
└── torch_cache/       # torch cache
```

创建命令：

```bash
mkdir -p /ssdcache/zizhaowang/vllm_lab/{envs,repos,models,wheelhouse,images,runs,logs,tmp,kv_cache_offload,offline_pkgs,hf_cache,vllm_cache,torch_cache}
```

---

## 2. 当前已经成功的基础环境

当前已经成功运行的环境是：

```text
环境名: vllm061-native
路径: /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
Python: 3.10.20
PyTorch: 2.4.0+cu121
torch.version.cuda: 12.1
vLLM: 0.6.1.post1
GPU: NVIDIA A100-PCIE-40GB
```

在 `comput10` 上已经验证：

```bash
source /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native/bin/activate

python - <<'PY'
import sys
print("python exe:", sys.executable)

import torch
import vllm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)
PY
```

输出确认：

```text
python exe: /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native/bin/python
torch: 2.4.0+cu121
torch cuda: 12.1
cuda available: True
gpu: NVIDIA A100-PCIE-40GB
vllm: 0.6.1.post1
```

这说明：

```text
1. 离线 conda-pack 环境可以在 comput10 上运行。
2. PyTorch 能识别 A100。
3. vLLM 0.6.1.post1 可以 import。
4. 当前不需要升级 comput10 的 CUDA / driver。
```

注意：`nvidia-smi` 中显示的 `CUDA Version: 12.2` 是 driver 支持的 CUDA runtime 能力，不等于必须安装系统 CUDA 12.2。当前 `torch 2.4.0+cu121` 自带 CUDA 12.1 用户态库，可以在 535 driver 上正常运行。

---

## 3. 已踩坑记录

### 3.1 comput10 不能联网

现象：

```bash
ping -c 3 223.5.5.5
```

输出：

```text
connect: Network is unreachable
```

原因：计算节点没有 default route，这是安全策略。

处理：所有下载在 admin 节点做，然后传到 `/ssdcache`。

---

### 3.2 comput10 上 yum 安装失败

现象：

```bash
sudo yum install -y nvme-cli
```

报错：

```text
Could not resolve host: mirrors.aliyun.com
No more mirrors to try
```

原因：comput10 无公网 / 无 DNS 外网解析。

处理：

```text
不要在 comput10 上 yum install。
如果确实需要系统包，在 admin 下载 rpm，再离线传到 comput10 安装。
```

---

### 3.3 不要升级 comput10 的 CUDA / driver

当前环境能跑：

```text
Driver 535.54.03
torch 2.4.0+cu121
vLLM 0.6.1.post1
A100
```

因此不要做：

```bash
sudo yum install cuda
sudo yum update nvidia-driver
sudo rpm -Uvh cuda-12-9...
```

原因：

```text
1. driver 是系统级组件，升级会影响整台共享 GPU 节点。
2. 升级 driver 通常需要重启节点。
3. 当前目标是 research 环境隔离，不是改系统。
4. 旧 CentOS 7 更大的风险是 glibc / wheel 兼容，而不是单纯 CUDA Toolkit。
```

---

### 3.4 vLLM 0.6.x 安装卡在 pyairports

失败命令：

```bash
python -m pip install --only-binary=:all: "vllm==0.6.1.post1"
```

或：

```bash
python -m pip install --only-binary=:all: "vllm==0.6.3.post1"
```

报错：

```text
outlines 0.0.46 depends on pyairports
outlines 0.0.45 depends on pyairports
outlines 0.0.44 depends on pyairports
outlines 0.0.43 depends on pyairports

No matching distributions available for pyairports
```

原因：

```text
vLLM 0.6.x -> outlines 0.0.43~0.0.46 -> pyairports
pyairports 没有满足 --only-binary=:all: 的 wheel
```

解决：

先在 admin 上把 `pyairports` 构建成本地 wheel：

```bash
python -m pip wheel \
    --no-deps \
    --wheel-dir ~/offline_pkgs/vllm061/wheelhouse \
    "pyairports==0.0.1"
```

然后安装 vLLM：

```bash
python -m pip install \
    --find-links ~/offline_pkgs/vllm061/wheelhouse \
    --only-binary=:all: \
    "vllm==0.6.1.post1"
```

---

### 3.5 conda-pack 报 pip 被覆盖

失败现象：

```bash
conda-pack -p ~/offline_pkgs/vllm061/env \
  -o ~/offline_pkgs/vllm061/vllm061_env_py310.tar.gz
```

报错：

```text
Files managed by conda were found to have been deleted/overwritten
- pip 26.0.1
```

原因：在 conda/micromamba 环境里执行过：

```bash
python -m pip install --upgrade pip setuptools wheel
```

这会让 pip 覆盖 conda 管理的 pip 文件。

正确原则：

```text
micromamba/conda 管：
- python
- pip
- setuptools
- wheel
- conda-pack

pip 管：
- vllm
- torch
- transformers
- huggingface_hub
- 其他 Python 包
```

不要用 pip 升级 conda 管理的核心工具包。

---

### 3.6 conda-unpack 语法错误

失败命令：

```bash
./bin/conda-unpack
```

报错：

```text
SyntaxError: invalid syntax
```

原因：`conda-unpack` 默认调用了系统 Python，CentOS 7 可能默认 Python 2，不支持 f-string。

正确命令：

```bash
cd /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
./bin/python ./bin/conda-unpack
```

---

## 4. GitHub 仓库应该怎么设计

你现在要做 research，不应该只在服务器上随便改 site-packages。建议分成两个层次：

```text
A. 你的研究控制仓库：kv-cache-offload-lab
B. 你的 vLLM 修改仓库：vllm-kv-offload 或 vLLM fork
```

### 4.1 推荐仓库一：kv-cache-offload-lab

这个仓库放你自己的实验脚本、环境构建脚本、Slurm 脚本、配置、结果摘要和文档。

建议结构：

```text
kv-cache-offload-lab/
├── README.md
├── MANIFEST.md
├── docs/
│   ├── 00_environment.md
│   ├── 01_offline_install_vllm061.md
│   ├── 02_model_download_and_transfer.md
│   ├── 03_experiment_protocol.md
│   └── 04_migration_guide.md
├── env/
│   ├── build_admin_vllm061.sh
│   ├── pack_admin_vllm061.sh
│   ├── activate_compute_vllm061.sh
│   ├── check_compute_env.sh
│   └── requirements.freeze.txt
├── scripts/
│   ├── download_model_admin.sh
│   ├── rsync_to_compute.sh
│   ├── run_smoke_vllm061.py
│   ├── run_latency_benchmark.py
│   └── collect_metrics.py
├── slurm/
│   ├── smoke_vllm061.slurm
│   ├── baseline_latency.slurm
│   └── offload_experiment.slurm
├── configs/
│   ├── models.yaml
│   ├── workloads.yaml
│   └── experiments.yaml
├── patches/
│   ├── README.md
│   └── vllm061_kv_trace.patch
├── results/
│   ├── README.md
│   └── summaries/
└── third_party/
    └── README.md
```

GitHub 里应该保存：

```text
- 脚本
- 配置
- 文档
- 小型结果摘要
- patch
- requirements.freeze.txt
- MANIFEST.md
```

GitHub 里不要保存：

```text
- conda-pack tar.gz
- 模型权重
- 大型实验 raw logs
- 大型 trace 文件
- vLLM cache
- KV offload 文件
```

大文件放在：

```text
/ssdcache/zizhaowang/vllm_lab/offline_pkgs
/ssdcache/zizhaowang/vllm_lab/models
/ssdcache/zizhaowang/vllm_lab/runs
```

GitHub 里只记录它们的路径、hash、生成脚本和说明。

---

### 4.2 推荐仓库二：vllm-kv-offload

如果你要深入改 vLLM，建议 fork 官方 vLLM，然后建立研究分支：

```text
fork: yourname/vllm-kv-offload
base: v0.6.1.post1
branch: kv-offload-v061
```

admin 上操作：

```bash
cd ~/offline_pkgs/vllm061
git clone https://github.com/vllm-project/vllm.git vllm-kv-offload
cd vllm-kv-offload
git checkout v0.6.1.post1
git checkout -b kv-offload-v061
```

如果你已经在 GitHub 上 fork 了，就改 remote：

```bash
git remote rename origin upstream
git remote add origin git@github.com:<yourname>/vllm-kv-offload.git
git push -u origin kv-offload-v061
```

如果暂时不想 fork，也可以先本地改，后面再推 GitHub。

---

## 5. 修改 vLLM 代码的现实路径

当前你有一个 wheel 安装好的环境：

```text
/ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
```

这个环境里的 vLLM 代码实际在：

```bash
python - <<'PY'
import vllm, os
print(os.path.dirname(vllm.__file__))
PY
```

通常类似：

```text
/ssdcache/zizhaowang/vllm_lab/envs/vllm061-native/lib/python3.10/site-packages/vllm
```

### 5.1 第一阶段：不要马上重编译 vLLM

你刚开始做 KV cache offloading，不建议第一步就改 C++/CUDA kernel。先分三步：

```text
Step 1: black-box baseline
Step 2: Python-level tracing / instrumentation
Step 3: offload policy prototype
```

原因：

```text
1. 当前 CentOS 7 + 离线节点源码编译 vLLM 会很麻烦。
2. 你现在的首要目标是理解 vLLM 的 KV cache 生命周期。
3. 先用 Python-level 改动验证策略，再决定是否碰 CUDA/C++。
```

---

### 5.2 最安全的改法：复制一个实验环境

不要直接改 `vllm061-native`。先复制一份：

```bash
cd /ssdcache/zizhaowang/vllm_lab/envs
cp -a vllm061-native vllm061-exp001
```

激活实验环境：

```bash
source /ssdcache/zizhaowang/vllm_lab/envs/vllm061-exp001/bin/activate
```

确认：

```bash
python - <<'PY'
import vllm, os
print(os.path.dirname(vllm.__file__))
PY
```

然后你可以改：

```text
/ssdcache/zizhaowang/vllm_lab/envs/vllm061-exp001/lib/python3.10/site-packages/vllm/
```

坏了就删：

```bash
rm -rf /ssdcache/zizhaowang/vllm_lab/envs/vllm061-exp001
```

再从 `vllm061-native` 复制一份。

---

### 5.3 研究代码应该怎么跟 GitHub 对齐

虽然你可以直接改 site-packages，但不要只把改动留在服务器里。正确做法是：

1. 在 `repos/vllm-kv-offload` 里做同样修改；
2. 用 git 管理；
3. 生成 patch；
4. 写脚本把 patch 应用到实验环境的 site-packages。

示例：

```bash
cd /ssdcache/zizhaowang/vllm_lab/repos/vllm-kv-offload
git diff > /ssdcache/zizhaowang/vllm_lab/repos/kv-cache-offload-lab/patches/vllm061_kv_trace.patch
```

或者在研究仓库中写：

```bash
scripts/apply_patch_to_env.sh
```

内容示例：

```bash
#!/bin/bash
set -euo pipefail

ENV_DIR=${1:-/ssdcache/zizhaowang/vllm_lab/envs/vllm061-exp001}
PATCH=${2:-/ssdcache/zizhaowang/vllm_lab/repos/kv-cache-offload-lab/patches/vllm061_kv_trace.patch}

SITE=$($ENV_DIR/bin/python - <<'PY'
import vllm, os
print(os.path.dirname(vllm.__file__))
PY
)

echo "Applying patch to $SITE"
cd "$SITE/.."
patch -p1 < "$PATCH"
```

这样迁移到新机器时，只需要：

```text
1. 重建 vllm061 环境
2. clone 你的 GitHub 仓库
3. apply patch
4. 跑同一个 slurm 脚本
```

---

## 6. vLLM 中应该先看哪些位置

做 KV cache offloading，先不要迷失在整个 vLLM 仓库。建议从这些入口理解：

```text
vllm/core/scheduler.py
    请求调度、prefill/decode 阶段、什么时候分配 KV block

vllm/core/block_manager.py
    逻辑/物理 KV block 管理，block table，free block，allocation

vllm/worker/cache_engine.py
    GPU/CPU cache 的创建、swap/copy 相关入口

vllm/worker/worker.py
    worker 执行模型、管理 cache engine

vllm/engine/llm_engine.py
    engine 主循环，调度器和 worker 的连接

vllm/attention/
    attention backend 与 KV cache 使用路径
```

不同 vLLM 版本文件名可能略有变化，以上以 v0.6.1 时代为主。你现在不要一开始就改 attention kernel，而应该先做：

```text
1. 打日志：请求何时分配 block
2. 打日志：block 什么时候被释放
3. 打日志：decode 阶段实际访问哪些 block
4. 统计：每轮 GPU KV cache 使用量
5. 统计：prefix/cache 命中与否
6. 统计：如果把旧 block offload，会影响哪些路径
```

---

## 7. 研究阶段规划

### 阶段 0：环境稳定性

目标：证明环境能稳定跑。

模型建议：

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
meta-llama/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
```

当前 `vllm061-native` 不建议直接跑 Gemma 4。Gemma 4 需要更新 vLLM/transformers 支持，应该单独走容器或新机器路线。

---

### 阶段 1：baseline

先跑不修改 vLLM 的 baseline：

```text
- GPU-only vLLM baseline
- 不同 input length
- 不同 output length
- 不同并发 batch
- 不同 max_model_len
```

指标：

```text
TTFT
TPOT
tokens/s
requests/s
GPU memory
CPU memory
本地 NVMe I/O
```

---

### 阶段 2：KV cache tracing

在 vLLM 中加入日志或 tracing：

```text
- block allocate/free
- GPU cache 使用量
- CPU cache 使用量
- 每个 sequence 的 block table
- prefill/decode 阶段切换
- scheduler 每轮 decision
```

先不要做真正 offload，只做观测。

---

### 阶段 3：CPU DRAM offloading prototype

先实现最简单策略：

```text
GPU HBM -> CPU DRAM
```

目标不是一开始追求性能，而是确认：

```text
1. 哪些 block 可以被迁移
2. decode 时如何找回来
3. scheduler 是否需要知道 block 状态
4. 是否影响正确性
```

---

### 阶段 4：local NVMe offloading prototype

再扩展到：

```text
GPU HBM -> CPU DRAM -> /ssdcache local NVMe
```

offload 目录固定：

```text
/ssdcache/zizhaowang/vllm_lab/kv_cache_offload
```

实验中不要用：

```text
/public
/data_test
/home
```

---

## 8. Slurm 和实验脚本建议

### 8.1 统一激活脚本

在 `comput10` 上保存：

```bash
/ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh
```

内容：

```bash
#!/bin/bash

export VLLM_LAB=/ssdcache/zizhaowang/vllm_lab

export HF_HOME=$VLLM_LAB/hf_cache
export HF_HUB_CACHE=$VLLM_LAB/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=$VLLM_LAB/hf_cache/hub
export TRANSFORMERS_CACHE=$VLLM_LAB/hf_cache/transformers

export VLLM_CACHE_ROOT=$VLLM_LAB/vllm_cache
export TORCH_HOME=$VLLM_LAB/torch_cache
export TMPDIR=$VLLM_LAB/tmp
export KV_OFFLOAD_DIR=$VLLM_LAB/kv_cache_offload

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TMPDIR" "$KV_OFFLOAD_DIR"

source $VLLM_LAB/envs/vllm061-native/bin/activate
```

---

### 8.2 环境检查脚本

放到 GitHub：

```bash
env/check_compute_env.sh
```

内容：

```bash
#!/bin/bash
set -euo pipefail

echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "python: $(which python)"
python --version

nvidia-smi

python - <<'PY'
import torch
import vllm
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)
PY

echo "Storage:"
findmnt -T /ssdcache/zizhaowang/vllm_lab
df -hT /ssdcache
```

---

### 8.3 smoke test Slurm 脚本

```bash
#!/bin/bash
#SBATCH -J vllm-smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -w comput10
#SBATCH -o /ssdcache/zizhaowang/vllm_lab/logs/vllm-smoke-%j.out

set -euo pipefail

source /ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh

bash /ssdcache/zizhaowang/vllm_lab/repos/kv-cache-offload-lab/env/check_compute_env.sh

CUDA_VISIBLE_DEVICES=0 python /ssdcache/zizhaowang/vllm_lab/repos/kv-cache-offload-lab/scripts/run_smoke_vllm061.py
```

---

## 9. 模型选择建议

### 9.1 当前 vllm061-native 推荐模型

推荐优先级：

```text
1. Qwen/Qwen2.5-7B-Instruct
2. Qwen/Qwen2.5-14B-Instruct
3. meta-llama/Llama-3.1-8B-Instruct
4. mistralai/Mistral-7B-Instruct-v0.3
```

原因：

```text
- 相对新
- 文本模型
- 比 Gemma 4 更容易被 vLLM 0.6.x 支持
- 单张 A100 40GB 更容易跑通
- 适合做 KV cache offloading baseline
```

### 9.2 Gemma 4 怎么办

Gemma 4 不建议用当前 vLLM 0.6.1 native 环境直接跑。

建议：

```text
Line 1: vllm061-native
    跑稳定 baseline、Qwen/Llama/Mistral、做 KV cache tracing/prototype

Line 2: gemma4-container
    单独使用 Apptainer/Singularity + 新 vLLM 镜像
    或等未来换新机器后用新 driver + 新 OS + 新 vLLM
```

也就是说：当前研究不要因为 Gemma 4 卡住。先用 Qwen2.5/Llama3.1 把 KV cache offloading pipeline 做起来。

---

## 10. 迁移到新机器时怎么做

迁移时不要只复制一个环境目录。应该复制这些东西：

```text
1. GitHub 仓库：kv-cache-offload-lab
2. vLLM fork：vllm-kv-offload
3. offline_pkgs 中的 conda-pack tar.gz
4. wheelhouse
5. models
6. MANIFEST.md
7. 关键实验配置 configs/*.yaml
8. Slurm 脚本
9. 结果摘要
```

新机器上第一步检查：

```bash
hostname
nvidia-smi
getconf GNU_LIBC_VERSION
df -hT
which apptainer || which singularity || true
```

然后判断：

```text
如果新机器仍然是老系统 / 老 driver：
    继续使用 vllm061-native 这条线

如果新机器是 Ubuntu 22.04/24.04 + 新 driver：
    可以考虑新 vLLM / Gemma 4 / 容器路线
```

迁移原则：

```text
研究代码靠 GitHub
环境靠 conda-pack / 容器
模型靠 rsync
实验复现靠 configs + MANIFEST + Slurm 脚本
```

---

## 11. 今天接下来应该做什么

你现在的下一步不是立刻改 offloading，而是把工程骨架立起来：

### Step 1：在 GitHub 建研究仓库

建议名：

```text
kv-cache-offload-lab
```

放：

```text
docs/
env/
scripts/
slurm/
configs/
patches/
results/
```

### Step 2：把当前成功环境记录进去

提交：

```text
MANIFEST.md
docs/00_environment.md
env/check_compute_env.sh
env/activate_compute_vllm061.sh
```

### Step 3：下载并部署 vLLM 0.6.1 源码

admin：

```bash
cd ~/offline_pkgs/vllm061
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.1.post1
cd ..
tar -czf vllm_repo_v0.6.1.post1.tar.gz vllm

rsync -avP vllm_repo_v0.6.1.post1.tar.gz \
  comput10:/ssdcache/zizhaowang/vllm_lab/offline_pkgs/
```

comput10：

```bash
cd /ssdcache/zizhaowang/vllm_lab/repos
tar -xzf /ssdcache/zizhaowang/vllm_lab/offline_pkgs/vllm_repo_v0.6.1.post1.tar.gz
```

### Step 4：下载一个 baseline 模型

先选：

```text
Qwen/Qwen2.5-7B-Instruct
```

admin 上下载，传到：

```text
/ssdcache/zizhaowang/vllm_lab/models/
```

### Step 5：跑 smoke test

确认：

```text
模型路径本地
vLLM 能 load
单 GPU 能 generate
日志写到 /ssdcache
```

### Step 6：开始做 tracing

先只加日志，不改策略：

```text
block allocation
block free
cache usage
prefill/decode phase
scheduler step
```

这一步完成后再进入真正 offloading 策略实现。

---

## 12. 核心结论

你的研究工程应该分成三条线：

```text
1. stable baseline line:
   vllm061-native，稳定可跑，用于 baseline 和早期 tracing。

2. modification line:
   vllm-kv-offload fork + patch 到实验环境，用于实现 offloading 策略。

3. migration/new-model line:
   容器 / 新机器 / 新 vLLM，用于 Gemma 4 等新模型。
```

不要把所有目标混在一个环境里。

当前最合理的策略是：

```text
先用 vLLM 0.6.1 + Qwen2.5/Llama3.1 把 KV cache offloading 研究 pipeline 做起来；
Gemma 4 以后单独用容器或新机器环境跑。
```

---

## References

- vLLM official GPU installation docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- vLLM v0.6.1 installation docs: https://docs.vllm.ai/en/v0.6.1/getting_started/installation.html
- vLLM GitHub repository: https://github.com/vllm-project/vllm
- vLLM Docker deployment docs: https://docs.vllm.ai/en/stable/deployment/docker/
- NVIDIA CUDA compatibility documentation: https://docs.nvidia.com/deploy/cuda-compatibility/
