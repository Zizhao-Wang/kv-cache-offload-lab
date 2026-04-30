# KV Cache Offloading Research 新人教程：Part I
# 从新机器到 vLLM 源码开发环境：Driver、micromamba、uv、CUDA/nvcc、FlashInfer 与模型测试
> 目标读者：刚开始做 LLM 推理 / vLLM / KV cache offloading 的同学。  
当前定位：这是**开发者版环境教程**，不是只跑模型的普通用户教程。  
核心目标：从一台新装 Ubuntu GPU 机器开始，搭好一个可以运行 vLLM、修改 vLLM 源码、支持 CUDA/nvcc JIT 编译、后续做 KV cache offloading 的研究环境。  
说明：本文使用 `$HOME`、`$LAB_ROOT` 等占位符，不绑定个人用户名。请根据实际机器替换路径。
>

---

# Part I-A：概念解释区
0–4 是概念解释。先搞清楚每个东西是什么、处在哪一层、为什么需要。不要一开始就把 Driver、CUDA、PyTorch、vLLM、FlashInfer、模型下载全部混在一起。

---

## 0. 这份 Part I 解决什么问题？
做 KV cache offloading research 时，最容易混乱的是这些东西全部混在一起：

```latex
NVIDIA Driver
CUDA Toolkit / nvcc
PyTorch CUDA wheel
vLLM
FlashAttention / FlashInfer / PagedAttention 等 attention 优化技术
Hugging Face 模型
KV cache
KV cache offloading
Docker / micromamba / uv
vLLM 源码开发
实验脚本
实验日志
本地 NVMe / 网络文件系统
GitHub 仓库
```

如果不分层，后面任何一个错误都会看起来像“整个环境坏了”。

Part I 的主线是：

```latex
先理解 LLM 推理链路
  ↓
再理解 GPU 软件栈层级
  ↓
安装 NVIDIA Driver，让 Linux 能用 GPU
  ↓
建立工作区和 GitHub 仓库边界
  ↓
用 micromamba 建 vLLM 开发环境
  ↓
用 uv 安装 vLLM 源码开发依赖
  ↓
在 micromamba 环境里补齐 CUDA/nvcc/dev headers
  ↓
editable 安装 vLLM 源码
  ↓
跑小模型 smoke test
  ↓
跑本地下载好的目标模型
  ↓
进入 vLLM 源码，做 KV cache offloading
```

注意：对普通“只跑模型”的用户，CUDA Toolkit / nvcc 可能可以晚点装；但对我们这种要做 vLLM / KV cache offloading 开发的人，建议在 `vllm-dev` 环境早期就把 `nvcc` 和必要 CUDA dev 组件装好，这样后面遇到 FlashInfer JIT、Triton/custom op、CUDA extension 时不会反复中断。

---

## 1. LLM 推理最小链路
一次普通的 LLM 推理大概是这样：

```latex
用户 prompt
  ↓
tokenizer 把文本变成 token ids
  ↓
推理引擎加载模型权重
  ↓
prefill 阶段：一次性处理输入 prompt，生成初始 KV cache
  ↓
decode 阶段：逐 token 生成输出，同时持续读取/追加 KV cache
  ↓
tokenizer 把输出 token ids 转回文本
```

这条链路里，每一段对应不同的系统问题：

```latex
tokenizer / chat template
  → 文本格式、对话模板、token ids

模型权重
  → safetensors、config、显存占用、加载路径

prefill
  → 处理 prompt，计算量大，attention 和矩阵乘法压力大

decode
  → 一个 token 一个 token 生成
  → 反复读取历史 KV cache
  → 对延迟、调度、KV cache 管理非常敏感

KV cache
  → 长上下文和高并发时最容易撑爆显存

vLLM
  → 管理请求、batch、KV block、调度、attention backend
```

对 KV cache offloading 来说，最核心的是这条路径：

```latex
GPU HBM 显存
  ↔ CPU DRAM 内存
  ↔ 本地 NVMe / SSD
```

不要把实验热路径放到：

```latex
GPU ↔ CPU ↔ 网络文件系统 / 分布式文件系统
```

否则测到的可能是网络存储延迟，而不是你的 offloading 策略。

---

## 2. GPU 推理环境的软件栈层级
可以把整个环境理解成一层一层搭起来的：

```latex
我们的研究代码
  └── KV cache offloading policy / benchmark / profiling

vLLM
  └── 请求调度、KV cache 管理、PagedAttention、OpenAI server

PyTorch / Triton / attention backend / vLLM kernels
  └── 调用 GPU 做矩阵计算、attention、KV cache 访问

CUDA runtime / cuDNN / NCCL 等用户态库
  └── 通常由 PyTorch wheel 或相关 Python 包提供

CUDA Toolkit / nvcc / CUDA headers
  └── 编译 CUDA/C++ extension、FlashInfer JIT、自定义 kernel 时需要

NVIDIA Driver
  └── Linux 控制 GPU 的系统驱动

GPU 硬件
  └── RTX 4090 / A100 / H100 等
```

最重要的结论：

```latex
NVIDIA Driver 是系统地基。
PyTorch CUDA wheel 是 Python 环境里的运行库。
CUDA Toolkit / nvcc 是开发工具箱。
vLLM 是我们后面要跑、要读、要改的推理系统。
FlashInfer / FlashAttention 等是 vLLM 可能调用的底层优化组件。
```

---

## 3. Driver、CUDA、PyTorch、vLLM 到底是什么关系？
### 3.1 NVIDIA Driver
Driver 是系统层组件。它让 Linux 能控制 NVIDIA GPU。

它提供：

```latex
/dev/nvidia*
nvidia-smi
GPU 任务提交能力
CUDA 程序运行所需的底层驱动支持
```

如果 driver 没装好：

```latex
PyTorch 看不到 GPU
vLLM 看不到 GPU
模型不能真正跑在 GPU 上
后面的所有问题都没有意义
```

所以第一步一定是：

```latex
先装 NVIDIA Driver，再验证 nvidia-smi。
```

### 3.2 nvidia-smi 里的 CUDA Version
`nvidia-smi` 里会显示类似：

```latex
CUDA Version: 13.2
```

这不等于系统已经安装了 CUDA Toolkit。它更准确地表示：

```latex
当前 NVIDIA Driver 支持的 CUDA runtime 能力上限。
```

它不等于：

```latex
已经安装 /usr/local/cuda
已经有 nvcc
PyTorch 必须使用完全一样的 CUDA 版本
```

### 3.3 PyTorch CUDA wheel
很多时候我们安装 PyTorch 时，PyTorch wheel 会自带或依赖 CUDA runtime、cuDNN、cuBLAS、NCCL 等用户态库。

例如可能看到：

```latex
torch: 2.11.0+cu130
torch.version.cuda: 13.0
```

这说明 PyTorch 使用的是 CUDA 13.0 相关 runtime。只要 driver 足够新，PyTorch 就可以使用 GPU。

验证方式：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

### 3.4 CUDA Toolkit / nvcc
CUDA Toolkit 是开发工具包，里面最关键的是：

```latex
nvcc
```

对普通推理用户，可能不需要第一天安装 `nvcc`；但对 vLLM / KV cache offloading 开发者，建议在 `vllm-dev` 环境里安装它，因为后面可能遇到：

```latex
FlashInfer JIT 编译
CUDA/C++ extension 编译
Triton / custom op 相关编译
vLLM csrc 修改
PagedAttention / attention backend 调试
```

注意：这里推荐安装到 **micromamba 环境内部**，而不是系统级 `/usr/local/cuda`。

推荐目标：

```latex
CUDA_HOME=$LAB_ROOT/envs/vllm-dev
nvcc=$LAB_ROOT/envs/vllm-dev/bin/nvcc
```

不要一上来执行：

```bash
sudo apt install nvidia-cuda-toolkit
sudo apt install cuda
```

那样容易污染系统级环境，也不利于迁移和复现。

---

## 4. vLLM、attention 优化技术和 KV cache offloading 的关系
vLLM 是推理系统，它负责：

```latex
加载模型
管理请求
continuous batching
管理 KV cache block
调用 attention backend
提供 OpenAI-compatible server
```

attention 优化技术是 vLLM 可能调用的底层优化组件，例如：

```latex
FlashAttention
PagedAttention
Triton attention kernels
FlashInfer
xFormers attention
vendor-specific optimized kernels
vLLM 自己维护的 attention backend
```

注意：

```latex
FlashAttention 只是 attention 优化技术中的一个例子。
FlashInfer 也只是某类高性能 kernel / sampler / JIT 组件。
它们不是唯一方案，也不是所有推理优化都叫 FlashAttention 或 FlashInfer。
```

对 KV cache offloading 来说，关注点是：

```latex
decode 阶段不断读取历史 KV cache。
当 KV cache 太大时，GPU 显存不够。
因此需要决定哪些 KV 留在 GPU，哪些放到 CPU DRAM 或 NVMe。
```

第一阶段通常先改 vLLM 的 Python / 系统逻辑层，例如：

```latex
scheduler
block manager
KV cache allocator
cache engine
prefix cache
swap / offload policy
profiling / logging
benchmark
```

底层 attention kernel / CUDA kernel 是后续阶段，等确定真的需要再处理。

---

# Part I-B：实操流程区
从这里开始是具体安装流程。  
这份版本按照开发者环境来写：**不仅跑模型，还要为后续 vLLM 源码开发、FlashInfer JIT、CUDA/nvcc 需求做准备。**

---

## 5. Step 1：检查系统和 GPU，安装 NVIDIA Driver
### 5.1 检查系统和 GPU
在新机器上先执行：

```bash
cat /etc/os-release
uname -a

lspci | grep -i nvidia || true
which nvidia-smi || true
nvidia-smi || true
```

如果 `lspci` 能看到 RTX 4090，但 `nvidia-smi` 不存在，说明：

```latex
GPU 硬件已经被系统识别。
但 NVIDIA Driver 还没有装好。
```

### 5.2 Ubuntu 上选择 driver
安装检测工具：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
```

查看推荐 driver：

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers list --gpgpu
```

如果输出里有类似：

```latex
nvidia-driver-595-open recommended
```

则安装：

```bash
sudo apt install -y nvidia-driver-595-open
sudo reboot
```

重启后验证：

```bash
nvidia-smi
```

期望看到：

```latex
NVIDIA GeForce RTX 4090
Driver Version: 595.xx
CUDA Version: 13.2
```

完成标准：

```latex
nvidia-smi 能看到所有 GPU。
```

如果这一步没成功，不要继续装 PyTorch / vLLM。

---

## 6. Step 2：建立工作区和 GitHub 仓库边界
### 6.1 工作区不是 GitHub 仓库
推荐使用一个物理工作区：

```bash
export LAB_ROOT=$HOME/vllm_lab
mkdir -p "$LAB_ROOT"/{envs,repos,models,wheelhouse,offline_pkgs,images,runs,logs,tmp,hf_cache,vllm_cache,torch_cache,kv_cache_offload}
```

这个目录是机器上的实验工作区，里面会有大文件和机器相关文件：

```latex
$LAB_ROOT/
├── envs/              # Python / micromamba 环境，不进 GitHub
├── models/            # 模型权重，不进 GitHub
├── logs/              # 日志，不进 GitHub
├── runs/              # 实验结果，不进 GitHub
├── hf_cache/          # Hugging Face cache，不进 GitHub
├── vllm_cache/        # vLLM cache，不进 GitHub
├── torch_cache/       # Torch cache，不进 GitHub
└── repos/             # GitHub 仓库放这里
```

### 6.2 vLLM 上游源码仓库和自己的 lab 仓库要分开
推荐：

```latex
$LAB_ROOT/repos/
├── vllm/                    # vLLM 上游源码仓库，后面可 fork
└── kv-cache-offload-lab/    # 你自己的 GitHub 仓库
```

自己的 GitHub 仓库放：

```latex
scripts/
configs/
docs/
benchmarks/
patches/
manifests/
notes/
README.md
.gitignore
```

不要把下面这些放进 GitHub：

```latex
models/
envs/
hf_cache/
vllm_cache/
torch_cache/
logs/
runs/
tmp/
*.safetensors
*.bin
*.pt
```

也不要把整个 vLLM 源码复制进自己的 lab 仓库。vLLM 源码应该作为独立仓库存在。后面如果要长期改 vLLM，应该 fork vLLM，然后把 `$LAB_ROOT/repos/vllm` 的 remote 指向你的 fork 或添加你的 fork remote。

---

## 7. Step 3：安装 micromamba 和 uv，创建 vLLM 开发环境
### 7.1 安装 micromamba
如果机器还没有 micromamba：

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

验证：

```bash
micromamba --version
```

### 7.2 创建 vLLM 开发环境
推荐明确放在 `$LAB_ROOT/envs/vllm-dev`：

```bash
export LAB_ROOT=$HOME/vllm_lab

micromamba create -y -p "$LAB_ROOT/envs/vllm-dev" -c conda-forge \
    python=3.12 \
    pip \
    setuptools \
    wheel \
    packaging \
    ninja \
    cmake \
    git

eval "$(micromamba shell hook --shell bash)"
micromamba activate "$LAB_ROOT/envs/vllm-dev"
```

验证：

```bash
which python
python --version
which pip
```

期望：

```latex
$LAB_ROOT/envs/vllm-dev/bin/python
Python 3.12.x
```

### 7.3 安装 uv
```bash
python -m pip install uv
uv --version
```

说明：

```latex
micromamba = 环境管理器，负责创建/激活 vllm-dev。
uv = Python 包安装工具，负责在 vllm-dev 里安装 vLLM、torch、transformers 等包。
```

它们不是两个虚拟环境。

---

## 8. Step 4：clone vLLM 源码并 editable 安装
### 8.1 clone vLLM
```bash
export LAB_ROOT=$HOME/vllm_lab
cd "$LAB_ROOT/repos"

git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### 8.2 editable 安装 vLLM
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
```

含义：

```latex
-e .
  editable install
  修改 Python 层源码后，不需要重新安装整个包。

VLLM_USE_PRECOMPILED=1
  先复用预编译 CUDA/C++ 组件。
  不一开始就完整编译 vLLM 底层 kernel。

--torch-backend=auto
  让 uv 根据当前机器和 driver 自动处理 torch backend。
```

如果国内网络下载超时，可以临时设置：

```bash
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

然后重试：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
```

验证：

```bash
python - <<'PY'
import sys
print("python:", sys.executable)

import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

import vllm
print("vllm:", vllm.__version__)
print("vllm file:", vllm.__file__)
PY
```

完成标准：

```latex
cuda available: True
vLLM 能 import
vllm file 指向 $LAB_ROOT/repos/vllm
```

---

## 9. Step 5：在 micromamba 环境里安装 CUDA/nvcc/dev 组件
这是开发者环境的重要步骤。目标不是系统级安装 CUDA，而是在 `vllm-dev` 里补齐编译工具。

### 9.1 先看 PyTorch CUDA 版本
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

如果看到：

```latex
torch.version.cuda: 13.0
```

就优先安装 CUDA 13.0 的开发组件。

### 9.2 安装 CUDA/nvcc 到 vllm-dev
```bash
export LAB_ROOT=$HOME/vllm_lab

micromamba install -y -p "$LAB_ROOT/envs/vllm-dev" -c nvidia \
    cuda-nvcc=13.0 \
    cuda-cudart-dev=13.0 \
    cuda-cccl=13.0
```

如果后面 FlashInfer JIT 报缺少 cuRAND 相关头文件或库，可以再安装：

```bash
micromamba install -y -p "$LAB_ROOT/envs/vllm-dev" -c nvidia \
    libcurand-dev
```

或根据搜索结果选择对应包：

```bash
micromamba search -c nvidia curand
```

### 9.3 验证 nvcc
```bash
export CUDA_HOME=$LAB_ROOT/envs/vllm-dev
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

which nvcc
nvcc --version
```

期望看到：

```latex
$LAB_ROOT/envs/vllm-dev/bin/nvcc
Cuda compilation tools, release 13.0
```

---

## 10. Step 6：写统一激活脚本
这个脚本应该放在自己的 GitHub 仓库里，例如：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh
```

创建：

```bash
export LAB_ROOT=$HOME/vllm_lab

cd "$LAB_ROOT/repos/kv-cache-offload-lab"
mkdir -p scripts/local_4090

cat > scripts/local_4090/activate_vllm_dev.sh <<'SCRIPT'
#!/bin/bash

export LAB_ROOT=$HOME/vllm_lab

# Activate micromamba env
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$LAB_ROOT/envs/vllm-dev"

# Cache / temp dirs
export HF_HOME=$LAB_ROOT/hf_cache
export HF_HUB_CACHE=$LAB_ROOT/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=$LAB_ROOT/hf_cache/hub
export TRANSFORMERS_CACHE=$LAB_ROOT/hf_cache/transformers

export VLLM_CACHE_ROOT=$LAB_ROOT/vllm_cache
export TORCH_HOME=$LAB_ROOT/torch_cache
export TMPDIR=$LAB_ROOT/tmp
export KV_OFFLOAD_DIR=$LAB_ROOT/kv_cache_offload

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TMPDIR" "$KV_OFFLOAD_DIR"

# CUDA Toolkit installed inside micromamba env, not system-wide.
export CUDA_HOME=$LAB_ROOT/envs/vllm-dev
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# PyPI NVIDIA CUDA headers/libs may live here.
# FlashInfer JIT may need these headers, e.g., curand.h.
export NVIDIA_CUDA_PYPI_ROOT="$CUDA_HOME/lib/python3.12/site-packages/nvidia/cu13"

if [ -d "$NVIDIA_CUDA_PYPI_ROOT/include" ]; then
  export CPATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${CPATH:-}"
  export C_INCLUDE_PATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
fi

if [ -d "$NVIDIA_CUDA_PYPI_ROOT/lib" ]; then
  export LIBRARY_PATH="$NVIDIA_CUDA_PYPI_ROOT/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$NVIDIA_CUDA_PYPI_ROOT/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "LAB_ROOT=$LAB_ROOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "NVIDIA_CUDA_PYPI_ROOT=$NVIDIA_CUDA_PYPI_ROOT"
SCRIPT

chmod +x scripts/local_4090/activate_vllm_dev.sh
```

以后进入环境：

```bash
source "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh"
```

验证：

```bash
which python
python --version
which nvcc
nvcc --version
echo "CUDA_HOME=$CUDA_HOME"
find "$CUDA_HOME" -name curand.h 2>/dev/null | head
```

---

## 11. Step 7：安装并测试 FlashInfer 相关依赖
vLLM 可能使用 FlashInfer 的 sampler 或 attention backend。如果不开启 FlashInfer sampler，vLLM 也可以正常跑模型；但开发者环境建议把它调通，避免后续遇到 JIT 问题。

### 11.1 安装 FlashInfer
```bash
source "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh"

export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

uv pip install flashinfer-python
```

如果清华源没有或版本不全，换官方 PyPI：

```bash
unset UV_DEFAULT_INDEX
uv pip install flashinfer-python
```

### 11.2 检查版本
```bash
python -m pip list | grep -i flashinfer
```

如果出现：

```latex
flashinfer-python  0.6.9
flashinfer-cubin   0.6.8.post1
```

说明版本不匹配，需要对齐：

```bash
uv pip install -U "flashinfer-cubin==0.6.9"
```

如果找不到 0.6.9，则降级成一致版本，例如：

```bash
uv pip install -U \
  "flashinfer-python==0.6.8.post1" \
  "flashinfer-cubin==0.6.8.post1"
```

### 11.3 常见 FlashInfer JIT 坑
#### 坑 1：没有 nvcc
现象：

```latex
nvcc: command not found
```

处理：

```latex
在 micromamba 环境里安装 cuda-nvcc。
不要优先 sudo apt install nvidia-cuda-toolkit。
```

#### 坑 2：没有 flashinfer
现象：

```latex
ModuleNotFoundError: No module named 'flashinfer'
```

处理：

```bash
uv pip install flashinfer-python
```

#### 坑 3：flashinfer-python 和 flashinfer-cubin 版本不一致
现象：

```latex
flashinfer-cubin version (...) does not match flashinfer version (...)
```

处理：

```latex
安装相同版本的 flashinfer-python 和 flashinfer-cubin。
```

#### 坑 4：找不到 curand.h
现象：

```latex
fatal error: curand.h: No such file or directory
```

可能原因：

```latex
curand.h 存在，但不在 nvcc 默认 include path 中。
```

检查：

```bash
find "$CUDA_HOME" -name curand.h 2>/dev/null
```

如果在：

```latex
$CUDA_HOME/lib/python3.12/site-packages/nvidia/cu13/include/curand.h
```

则需要把该路径加入 `CPATH` / `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH`，也就是上面激活脚本里的 `NVIDIA_CUDA_PYPI_ROOT` 相关设置。

---

## 12. Step 8：跑小模型 smoke test
建议把测试脚本放到自己的 lab 仓库：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab/scripts/test_flashinfer_sampler_qwen_small.py
```

创建：

```bash
cd "$LAB_ROOT/repos/kv-cache-offload-lab"

cat > scripts/test_flashinfer_sampler_qwen_small.py <<'PY'
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        disable_log_stats=True,
        enforce_eager=True,
    )

    outputs = llm.generate(
        ["用一句话解释什么是 KV cache offloading。"],
        SamplingParams(max_tokens=64, temperature=0.0),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
cd "$LAB_ROOT/repos/vllm"

source "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh"

rm -rf ~/.cache/flashinfer
rm -rf ~/.cache/torch_extensions

CUDA_VISIBLE_DEVICES=0 python "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/test_flashinfer_sampler_qwen_small.py"
```

完成标准：

```latex
模型能加载。
vLLM engine 能初始化。
能生成一小段文本。
FlashAttention / FlashInfer 相关路径不再因为缺依赖而失败。
```

注意：如果激活脚本里设置了 `VLLM_USE_FLASHINFER_SAMPLER=0`，日志会显示使用 PyTorch-native sampler。如果要测试 FlashInfer sampler，需要确保没有设置这个变量：

```bash
unset VLLM_USE_FLASHINFER_SAMPLER
```

---

## 13. Step 9：下载并运行目标模型
### 13.1 下载模型
```bash
source "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh"

hf download Qwen/<your-qwen-model> \
  --local-dir "$LAB_ROOT/models/<your-qwen-model>" \
  --max-workers 8
```

后台下载：

```bash
mkdir -p "$LAB_ROOT/logs"

nohup hf download Qwen/<your-qwen-model> \
  --local-dir "$LAB_ROOT/models/<your-qwen-model>" \
  --max-workers 8 \
  > "$LAB_ROOT/logs/download_qwen.log" 2>&1 &
```

检查：

```bash
du -sh "$LAB_ROOT/models/<your-qwen-model>"
find "$LAB_ROOT/models/<your-qwen-model>" -maxdepth 1 -type f | sort
find "$LAB_ROOT/models" -type f \( -name "*.incomplete" -o -name "*.tmp" -o -name "*.lock" \) -print
```

### 13.2 运行本地模型
不要用 `python - <<'PY'` 跑大模型测试。如果 vLLM 使用 multiprocessing spawn，stdin 脚本会触发：

```latex
FileNotFoundError: ... <stdin>
```

应该写成真实 `.py` 文件，并使用：

```python
if __name__ == "__main__":
    main()
```

示例脚本：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab/scripts/run_qwen_local.py
```

核心结构：

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    model_path = "/path/to/local/model"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    messages = [
        {"role": "user", "content": "用一句话解释什么是 KV cache offloading。"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=2,
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_log_stats=True,
    )

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=64, temperature=0.0),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python "$LAB_ROOT/repos/kv-cache-offload-lab/scripts/run_qwen_local.py"
```

---

## 14. Step 10：进入 vLLM 源码阅读和 KV cache offloading 修改
模型跑通后，再进入源码开发。

刚开始不要直接改 CUDA kernel。建议先看 Python / 系统逻辑层：

```latex
request scheduling
sequence lifecycle
block allocation
KV block manager
prefix cache
swap / offload policy
memory accounting
profiling / logging
benchmark
```

先回答：

```latex
一个 request 进入 vLLM 后，在哪里排队？
什么时候分配 KV block？
KV block 和 token 的关系是什么？
什么时候释放 KV block？
prefix cache 怎么复用？
显存不足时 vLLM 现在怎么处理？
是否已有 swap/offload 相关路径？
```

之后如果确实需要，再进入：

```latex
csrc/
CUDA kernel
PagedAttention kernel
attention backend
Triton kernel
torch custom op
```

---

## 15. 第一阶段完成标准
开发者环境第一阶段完成标准：

```latex
1. nvidia-smi 能看到所有 GPU。
2. $LAB_ROOT 工作区已建立。
3. $LAB_ROOT/repos/kv-cache-offload-lab 是自己的 GitHub 仓库。
4. $LAB_ROOT/repos/vllm 是独立 vLLM 源码仓库。
5. micromamba / vllm-dev 环境能正常激活。
6. uv 可用。
7. vLLM editable install 成功。
8. PyTorch 能看到 GPU。
9. nvcc 在 vllm-dev 环境里可用。
10. CUDA_HOME / CUDA_PATH 指向 vllm-dev。
11. FlashInfer 相关 Python 包和 cubin 版本一致。
12. FlashInfer JIT 需要的 CUDA headers/libs 路径已配置。
13. 小模型 smoke test 能成功。
14. 本地目标模型能完成最小推理。
15. 明确 baseline 运行、vLLM 源码修改、CUDA kernel 修改是三个不同阶段。
```

---

# Part I 小结
Part I 的逻辑是：

```latex
0–4：概念解释
  推理链路、GPU 软件栈、Driver/CUDA/PyTorch/vLLM/KV cache/attention backend 的关系。

5–7：基础系统和 Python 环境
  安装 NVIDIA Driver，建立工作区，安装 micromamba 和 uv。

8–11：vLLM 开发者环境
  clone vLLM，editable install，安装 CUDA/nvcc/dev 组件，配置 activation 脚本，修 FlashInfer JIT 依赖。

12–13：模型运行测试
  先跑小模型，再跑目标本地模型。

14：源码开发方向
  先改 Python 层 KV cache 管理逻辑，后面必要时再进入 CUDA kernel。

15：完成标准
  判断新人是否完成第一阶段开发环境搭建。
```

一句话：

```latex
我们不是简单地“装一个大模型环境”，而是在搭建一个可以长期开发 vLLM / KV cache offloading 的 GPU 系统环境。Driver 是系统地基，micromamba 是环境边界，uv 是包安装工具，CUDA/nvcc 装在 vllm-dev 里，vLLM 源码和自己的 lab 仓库分开，模型和日志不进 GitHub。
```

