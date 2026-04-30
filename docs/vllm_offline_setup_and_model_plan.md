# vLLM 离线环境搭建记录、踩坑总结与后续模型方案

> 适用场景：集群有公网管理节点 `admin`，但 GPU 计算节点 `comput10` 按安全策略不能联网。  
> 当前目标：为 KV cache offloading 研究搭建一个隔离、可删除、可迁移、可复现的 vLLM 推理环境。

---

## 1. 当前集群与硬件背景

### 1.1 节点角色

当前集群至少有两类节点：

```text
admin：
- 管理 / 登录节点
- 可以访问公网
- 用于下载 GitHub repo、Python wheel、模型权重、容器镜像、RPM 包
- 不建议跑正式 GPU 实验

comput10：
- GPU 计算节点
- 无公网路由，管理员明确要求不要接公网
- 用于离线运行推理实验
- 所有代码、模型、cache、实验输出都应该放在本地 NVMe
```

### 1.2 comput10 当前系统

```text
OS: CentOS Linux 7.9.2009 (Core)
Kernel: 3.10.0-1160.49.1.el7.x86_64
GPU: NVIDIA A100-PCIE-40GB x 4
NVIDIA Driver: 535.54.03
nvidia-smi 显示 CUDA Version: 12.2
```

`comput10` 没有 default route：

```bash
ip route
```

输出中只有类似：

```text
10.10.0.0/16 dev ens31f0 ...
10.11.1.0/28 dev ens31f1 ...
12.12.12.0/24 dev ib0 ...
172.17.0.0/16 dev docker0 ...
```

没有：

```text
default via ...
```

所以在 `comput10` 上执行：

```bash
ping -c 3 223.5.5.5
```

会出现：

```text
connect: Network is unreachable
```

这不是故障，而是安全策略。

---

## 2. 存储路径选择

### 2.1 comput10 上的文件系统

已经观察到：

```text
/ssdcache      -> /dev/nvme1n1, ext4, 本地 NVMe
/mnt/nvme0n1   -> /dev/nvme0n1p1, ext4, 本地 NVMe
/mnt/nvme2n1   -> /dev/nvme2n1, ext4, 本地 NVMe
/public        -> nfs4，网络文件系统
/data_test     -> parastor，分布式/并行文件系统
/home          -> root 盘，不建议放大量实验数据
/tmp           -> root 盘，不建议作为 KV offload 热路径
```

### 2.2 推荐路径

后续统一使用：

```text
/ssdcache/zizhaowang/vllm_lab/
```

目录结构：

```text
/ssdcache/zizhaowang/vllm_lab/
├── envs/              # 离线 Python 环境
├── repos/             # vLLM 源码
├── models/            # 模型权重
├── wheelhouse/        # 离线 wheel 包
├── images/            # Apptainer/Singularity 容器镜像
├── runs/              # 每次实验输出
├── logs/              # Slurm / 服务日志
├── tmp/               # 临时文件
├── kv_cache_offload/  # KV cache disk offloading 目录
├── offline_pkgs/      # admin 传来的离线包
├── hf_cache/          # Hugging Face cache
├── vllm_cache/        # vLLM cache
└── torch_cache/       # torch cache
```

在 `comput10` 上创建：

```bash
mkdir -p /ssdcache/zizhaowang/vllm_lab/{envs,repos,models,wheelhouse,images,runs,logs,tmp,kv_cache_offload,offline_pkgs,hf_cache,vllm_cache,torch_cache}
```

确认路径在本地 NVMe：

```bash
findmnt -T /ssdcache/zizhaowang/vllm_lab
df -hT /ssdcache
```

---

## 3. 总体安装原则

### 3.1 comput10 上不要做的事情

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

原因：

1. `comput10` 本来就没有公网。
2. 管理员明确要求计算节点不要接触外网。
3. 研究环境应该隔离、可删除、可复现。
4. 所有下载和构建都应该在 `admin` 上完成。

### 3.2 正确流程

```text
admin 下载 / 构建 / 打包
        ↓
rsync/scp 传到 comput10
        ↓
comput10 离线解包
        ↓
comput10 离线运行
```

---

## 4. 当前已经成功的 native 环境

当前成功环境：

```text
环境名：vllm061-native
路径：/ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
Python：3.10.20
PyTorch：2.4.0+cu121
torch.version.cuda：12.1
vLLM：0.6.1.post1
GPU：NVIDIA A100-PCIE-40GB
```

在 `comput10` 上验证：

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

成功输出：

```text
python exe: /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native/bin/python
torch: 2.4.0+cu121
torch cuda: 12.1
cuda available: True
gpu: NVIDIA A100-PCIE-40GB
vllm: 0.6.1.post1
```

结论：

```text
1. 离线 conda-pack 环境可以在 comput10 上运行。
2. PyTorch cu121 wheel 可以在 Driver 535.54.03 上正常使用。
3. vLLM 0.6.1.post1 可以正常 import。
4. 不需要升级 comput10 的系统 CUDA / NVIDIA driver。
```

---

## 5. 已踩坑总结

### 5.1 坑一：计算节点没有网络

现象：

```bash
ping -c 3 223.5.5.5
```

输出：

```text
connect: Network is unreachable
```

解释：

`comput10` 没有 default route，这是安全策略，不应修改。

处理方式：

```text
所有联网下载都在 admin 上完成。
comput10 只接收离线包并运行实验。
```

---

### 5.2 坑二：`sudo yum install nvme-cli` 失败

现象：

```bash
sudo yum install -y nvme-cli
```

报错：

```text
Could not resolve host: mirrors.aliyun.com
No more mirrors to try
```

原因：

`comput10` 无法访问外网 yum 源。

处理：

```text
不要在 comput10 上 yum install。
如果确实需要系统包，应在 admin 上下载 RPM，再离线传到 comput10 安装。
```

但本次 vLLM 实验不依赖 `nvme-cli`，因为已经通过：

```bash
df -hT
lsblk
findmnt -T /ssdcache
```

确认了 `/ssdcache` 是本地 NVMe。

---

### 5.3 坑三：不要升级 comput10 的 CUDA / driver

当前 `nvidia-smi` 显示：

```text
Driver Version: 535.54.03
CUDA Version: 12.2
```

这里的 `CUDA Version: 12.2` 主要表示当前 NVIDIA driver 暴露的 CUDA runtime capability，不等于必须安装 `/usr/local/cuda-12.2`，也不等于 vLLM 必须使用系统 CUDA Toolkit 12.2。

本次成功环境使用的是：

```text
torch 2.4.0+cu121
torch.version.cuda = 12.1
```

它依然可以在当前 A100 + Driver 535.54.03 上运行。

原因是 CUDA 11 以后支持同 major 版本内的 minor-version compatibility。CUDA 12.x 应用通常要求 driver 属于 525+ 且低于 580 的兼容范围；但如果应用依赖更新 driver 特性、PTX JIT 或新 CUDA 13，则可能需要更新 driver。

本环境的结论：

```text
vLLM 0.6.1 + PyTorch cu121 可以跑。
不需要升级系统 CUDA。
不需要升级 NVIDIA driver。
不要在共享计算节点上动 driver。
```

---

### 5.4 坑四：`vllm==0.6.3.post1` / `0.6.1.post1` 卡在 pyairports

原始命令：

```bash
python -m pip install --only-binary=:all: "vllm==0.6.3.post1"
```

或：

```bash
python -m pip install --only-binary=:all: "vllm==0.6.1.post1"
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
vLLM 0.6.x 依赖 outlines 0.0.43~0.0.46。
这些 outlines 版本依赖 pyairports。
pyairports 没有可直接满足 --only-binary=:all: 的 wheel。
```

修复：

在 `admin` 上先把 `pyairports==0.0.1` 构建成本地 wheel：

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

### 5.5 坑五：`conda-pack` 报 pip 文件被覆盖

错误：

```text
CondaPackError:
Files managed by conda were found to have been deleted/overwritten

- pip 26.0.1:
    lib/python3.10/site-packages/pip-26.0.1.dist-info/INSTALLER
    ...
```

原因：

在 conda/micromamba 环境中执行了：

```bash
python -m pip install --upgrade pip setuptools wheel
```

这会让 pip 覆盖 conda 管理的文件，导致 `conda-pack` 检测到环境不一致。

正确做法：

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
- 其他 Python 依赖
```

不要在 conda/micromamba 环境里用 pip 升级 pip 本身。

修复方式：

```bash
micromamba deactivate
rm -rf ~/offline_pkgs/vllm061/env
rm -rf ~/offline_pkgs/vllm061/wheelhouse
mkdir -p ~/offline_pkgs/vllm061/wheelhouse
```

然后重新创建干净环境。

---

### 5.6 坑六：`./bin/conda-unpack` 报 f-string 语法错误

在 `comput10` 上直接执行：

```bash
./bin/conda-unpack
```

报错：

```text
SyntaxError: invalid syntax
```

原因：

`conda-unpack` 可能走了 CentOS 7 的系统 Python 2，而 Python 2 不支持 f-string。

正确做法：

```bash
cd /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
./bin/python ./bin/conda-unpack
```

之后激活环境：

```bash
source /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native/bin/activate
```

---

## 6. 成功安装 vLLM 0.6.1 的完整流程

下面是整理后的标准流程。

### 6.1 在 admin 上创建干净环境

```bash
mkdir -p ~/offline_pkgs/vllm061/wheelhouse
cd ~/offline_pkgs/vllm061

eval "$(micromamba shell hook --shell bash)"

micromamba create -y -p ~/offline_pkgs/vllm061/env \
    -c conda-forge \
    python=3.10 \
    pip \
    setuptools \
    wheel \
    packaging \
    build \
    conda-pack

micromamba activate ~/offline_pkgs/vllm061/env
```

注意：不要执行 `python -m pip install --upgrade pip setuptools wheel`。

### 6.2 构建 pyairports wheel

```bash
python -m pip wheel \
    --no-deps \
    --wheel-dir ~/offline_pkgs/vllm061/wheelhouse \
    "pyairports==0.0.1"
```

检查：

```bash
ls -lh ~/offline_pkgs/vllm061/wheelhouse
```

应该看到：

```text
pyairports-0.0.1-py3-none-any.whl
```

### 6.3 安装 vLLM

```bash
python -m pip install \
    --find-links ~/offline_pkgs/vllm061/wheelhouse \
    --only-binary=:all: \
    "vllm==0.6.1.post1"
```

检查：

```bash
python - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
PY
```

检查依赖：

```bash
python -m pip check
```

导出依赖：

```bash
python -m pip freeze > ~/offline_pkgs/vllm061/requirements.freeze.txt
```

### 6.4 打包环境

```bash
conda-pack -p ~/offline_pkgs/vllm061/env \
    -o ~/offline_pkgs/vllm061/vllm061_env_py310.tar.gz
```

### 6.5 传到 comput10

```bash
ssh comput10 "mkdir -p /ssdcache/zizhaowang/vllm_lab/{envs,repos,offline_pkgs,models,logs,runs,tmp,kv_cache_offload,wheelhouse}"

rsync -avP ~/offline_pkgs/vllm061/vllm061_env_py310.tar.gz \
    comput10:/ssdcache/zizhaowang/vllm_lab/offline_pkgs/

rsync -avP ~/offline_pkgs/vllm061/requirements.freeze.txt \
    comput10:/ssdcache/zizhaowang/vllm_lab/offline_pkgs/

rsync -avP ~/offline_pkgs/vllm061/wheelhouse/ \
    comput10:/ssdcache/zizhaowang/vllm_lab/wheelhouse/
```

### 6.6 在 comput10 上解包

```bash
mkdir -p /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native
cd /ssdcache/zizhaowang/vllm_lab/envs/vllm061-native

tar -xzf /ssdcache/zizhaowang/vllm_lab/offline_pkgs/vllm061_env_py310.tar.gz
./bin/python ./bin/conda-unpack
```

激活并验证：

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

---

## 7. 下载 vLLM 0.6.1 源码

这一步还需要继续做。

### 7.1 在 admin 上下载源码

```bash
mkdir -p ~/offline_pkgs/vllm061
cd ~/offline_pkgs/vllm061

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.1.post1
cd ..

tar -czf vllm_repo_v0.6.1.post1.tar.gz vllm
```

### 7.2 传到 comput10

```bash
ssh comput10 "mkdir -p /ssdcache/zizhaowang/vllm_lab/{repos,offline_pkgs}"

rsync -avP ~/offline_pkgs/vllm061/vllm_repo_v0.6.1.post1.tar.gz \
    comput10:/ssdcache/zizhaowang/vllm_lab/offline_pkgs/
```

### 7.3 在 comput10 上解压

```bash
cd /ssdcache/zizhaowang/vllm_lab/repos

tar -xzf /ssdcache/zizhaowang/vllm_lab/offline_pkgs/vllm_repo_v0.6.1.post1.tar.gz
```

检查：

```bash
cd /ssdcache/zizhaowang/vllm_lab/repos/vllm
git status
git rev-parse --short HEAD
```

---

## 8. 建立统一激活脚本

在 `comput10` 上执行：

```bash
cat > /ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh <<'EOF'
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
EOF

chmod +x /ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh
```

以后每次使用：

```bash
source /ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh
```

---

## 9. 当前环境能不能跑 Gemma 4？

结论：

```text
当前 vllm061-native 环境不能可靠运行 Gemma 4。
```

原因不是 A100 不够，也不是 CUDA 12.1 / 12.2 本身不能跑，而是**软件栈太旧**：

```text
当前环境：
- vLLM 0.6.1.post1
- transformers 版本较旧
- PyTorch 2.4.0+cu121
- CentOS 7 原生环境

Gemma 4 需要：
- vLLM 新版本里的 Gemma4ForConditionalGeneration / Gemma4ForCausalLM 支持
- transformers 5.5.x 级别支持 gemma4 model_type
- 更现代的 vLLM runtime / container 环境
```

vLLM 最新文档中已经有 `Gemma4ForConditionalGeneration` 支持，vLLM release v0.19.1 明确提到升级 Transformers v5.5.3 并修复多项 Gemma 4 bug。Google 官方文档显示 Gemma 4 包含 E2B、E4B、31B、26B-A4B 等版本，并且 E2B/E4B 的 BF16 显存需求大约分别为 9.6GB 和 15GB，所以从**硬件容量**看，A100 40GB 可以跑 E2B/E4B；但从**当前 native 软件栈**看，vLLM 0.6.1 不适合直接跑 Gemma 4。

### 9.1 重要逻辑：为什么 CUDA 12.1 能跑，但 Gemma 4 不能跑？

这两个问题不是一回事：

```text
问题 A：当前 torch/vLLM 能不能用 GPU？
答案：能。torch 2.4.0+cu121 已经在 A100 上验证成功。

问题 B：当前 vLLM 0.6.1 知不知道 Gemma 4 这个模型结构？
答案：大概率不知道，也不应该强行跑。

问题 C：Gemma 4 小模型的显存够不够？
答案：E2B/E4B 显存角度够，但需要新 vLLM / 新 transformers / 推荐容器。

问题 D：是否必须升级 comput10 系统 CUDA？
答案：不应该。要用容器或单独新环境隔离，而不是升级共享节点系统 CUDA/driver。
```

---

## 10. 如果必须跑 Gemma 4，推荐方案

### 10.1 首选：容器路线

对 Gemma 4，不建议在 CentOS 7 native 环境硬装新 vLLM。推荐：

```text
admin 下载 / 构建 vLLM Gemma 4 容器
        ↓
传 .sif 到 comput10
        ↓
comput10 用 Apptainer/Singularity --nv 离线运行
```

先检查 `comput10`：

```bash
which apptainer || which singularity || true
```

如果没有，需要让管理员提供 Apptainer/Singularity。

在 `admin` 或其他联网构建机上构建：

```bash
mkdir -p ~/offline_pkgs/gemma4/images
cd ~/offline_pkgs/gemma4/images

apptainer build vllm_gemma4.sif docker://vllm/vllm-openai:gemma4
```

如果官方 tag 拉取失败，需要根据当前 vLLM Docker Hub tag 选择 CUDA 12.x 的 Gemma 4 镜像。避免选 CUDA 13 镜像，因为 CUDA 13.x 的最低 driver 范围是 580+，而当前 `comput10` driver 是 535。

传到 `comput10`：

```bash
rsync -avP ~/offline_pkgs/gemma4/images/vllm_gemma4.sif \
    comput10:/ssdcache/zizhaowang/vllm_lab/images/
```

### 10.2 推荐先跑 Gemma 4 E2B 或 E4B

不要直接跑 31B 或 26B-A4B。

推荐优先级：

```text
1. google/gemma-4-E2B-it
   - 更小，更适合第一轮验证
   - 对 A100 40GB 余量更大

2. google/gemma-4-E4B-it
   - 新模型，能力更强
   - BF16 权重约 15GB，A100 40GB 显存可承受，但要控制 max_model_len

暂不建议：
- google/gemma-4-31B-it
- google/gemma-4-26B-A4B-it

原因：
- BF16 权重本身大约 48GB/58GB 级别，单张 A100 40GB 不适合
- 需要量化或多卡 tensor parallel
- 更容易把实验问题复杂化
```

### 10.3 admin 上下载模型

```bash
mkdir -p ~/offline_pkgs/gemma4/models

huggingface-cli login

huggingface-cli download google/gemma-4-E2B-it \
  --local-dir ~/offline_pkgs/gemma4/models/google_gemma-4-E2B-it \
  --local-dir-use-symlinks False
```

或者：

```bash
huggingface-cli download google/gemma-4-E4B-it \
  --local-dir ~/offline_pkgs/gemma4/models/google_gemma-4-E4B-it \
  --local-dir-use-symlinks False
```

传到 `comput10`：

```bash
rsync -avP ~/offline_pkgs/gemma4/models/google_gemma-4-E2B-it \
  comput10:/ssdcache/zizhaowang/vllm_lab/models/
```

### 10.4 comput10 上运行 Gemma 4 容器

```bash
export VLLM_LAB=/ssdcache/zizhaowang/vllm_lab
export GEMMA4_IMAGE=$VLLM_LAB/images/vllm_gemma4.sif
export GEMMA4_MODEL=$VLLM_LAB/models/google_gemma-4-E2B-it
```

检查容器 GPU：

```bash
apptainer exec --nv \
  --bind $VLLM_LAB:/workspace \
  $GEMMA4_IMAGE \
  nvidia-smi
```

启动服务：

```bash
apptainer exec --nv \
  --bind $VLLM_LAB:/workspace \
  --env HF_HOME=/workspace/hf_cache \
  --env HF_HUB_CACHE=/workspace/hf_cache/hub \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_OFFLINE=1 \
  --env VLLM_CACHE_ROOT=/workspace/vllm_cache \
  --env TMPDIR=/workspace/tmp \
  $GEMMA4_IMAGE \
  vllm serve /workspace/models/google_gemma-4-E2B-it \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt image=0,audio=0
```

说明：

```text
--max-model-len 8192
    第一轮先控制上下文长度，避免 KV cache 把显存打满。

--gpu-memory-utilization 0.85
    给系统和 profiling 留余量。

--limit-mm-per-prompt image=0,audio=0
    如果只做文本推理，禁用图像/音频输入预算，减少多模态 profiling 开销。
```

---

## 11. 如果不跑 Gemma 4，当前环境推荐跑什么模型？

当前 `vllm061-native` 更适合跑以下模型作为 KV cache offloading baseline。

### 11.1 首选：Qwen2.5 系列

推荐：

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
```

原因：

```text
1. 相对新，仍然适合研究展示。
2. 文本模型，少多模态依赖。
3. 架构相对稳定，vLLM 0.6.x 更容易支持。
4. 7B 单卡 A100 40GB 余量大，适合做长上下文和 KV cache 实验。
5. 14B 也可以尝试，但需要更谨慎控制 context length 和 batch size。
```

建议第一轮：

```text
Qwen2.5-7B-Instruct
max_model_len = 8192 / 16384
batch size 从 1 开始
output length 从 128 / 512 开始
```

### 11.2 次选：Llama 3.1 8B

推荐：

```text
meta-llama/Llama-3.1-8B-Instruct
```

原因：

```text
1. 模型生态成熟。
2. vLLM 支持较稳定。
3. 适合做 baseline。
```

注意：

```text
需要 Hugging Face gated access。
admin 上登录 HF 后下载。
```

### 11.3 次选：Mistral 7B Instruct v0.3

推荐：

```text
mistralai/Mistral-7B-Instruct-v0.3
```

原因：

```text
1. 模型较小，跑通容易。
2. 适合调试 benchmark pipeline。
3. 适合作为非 Qwen 的对照 baseline。
```

### 11.4 不建议在当前 native 环境硬跑的模型

```text
Gemma 4
Qwen3 / Qwen3.5 / Kimi K2.5 等非常新的模型
大型 MoE 模型
需要最新 transformers / 最新 vLLM 才支持的模型
```

原因：

```text
当前 vLLM 0.6.1 太老。
CentOS 7 原生环境太老。
强行升级容易进入依赖地狱。
```

---

## 12. 推荐研究路线

### 12.1 当前 native 环境路线

用于稳定 baseline：

```text
vllm061-native
├── Qwen2.5-7B-Instruct
├── Qwen2.5-14B-Instruct
├── Llama-3.1-8B-Instruct
└── Mistral-7B-Instruct-v0.3
```

目标：

```text
1. 跑通 vLLM 推理。
2. 跑 TTFT / TPOT / throughput。
3. 记录 GPU memory / CPU memory / NVMe I/O。
4. 建立 KV cache 实验 workload。
5. 阅读 vLLM 0.6.1 源码和 block manager。
```

### 12.2 新模型 / Gemma 4 容器路线

用于追新模型：

```text
gemma4-container
├── google/gemma-4-E2B-it
└── google/gemma-4-E4B-it
```

目标：

```text
1. 不污染 CentOS 7 native 环境。
2. 通过 Apptainer/Singularity 隔离用户态库。
3. 方便未来迁移到新机器。
4. 如果要用 CUDA 13 镜像，必须换更新 driver 的机器。
```

### 12.3 不建议现在做的事情

```text
1. 在 comput10 上升级 driver。
2. 在 comput10 上安装系统 CUDA。
3. 在 comput10 上打开外网。
4. 在当前 vllm061-native 中硬升级到最新 vLLM。
5. 在 CentOS 7 native 环境里硬装 Gemma 4 依赖。
```

---

## 13. 后续推荐执行顺序

### Step 1：完成 vLLM 0.6.1 源码下载

按第 7 节操作。

### Step 2：在 admin 上下载 Qwen2.5-7B-Instruct

```bash
mkdir -p ~/offline_pkgs/models

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ~/offline_pkgs/models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

传到 `comput10`：

```bash
rsync -avP ~/offline_pkgs/models/Qwen2.5-7B-Instruct \
  comput10:/ssdcache/zizhaowang/vllm_lab/models/
```

### Step 3：在 comput10 上跑 Qwen2.5 最小推理

```bash
source /ssdcache/zizhaowang/vllm_lab/activate_vllm061.sh

CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm import LLM, SamplingParams

model_path = "/ssdcache/zizhaowang/vllm_lab/models/Qwen2.5-7B-Instruct"

llm = LLM(
    model=model_path,
    dtype="float16",
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
)

outputs = llm.generate(
    ["Explain KV cache offloading in one paragraph."],
    SamplingParams(max_tokens=128, temperature=0.0),
)

for o in outputs:
    print(o.outputs[0].text)
PY
```

### Step 4：再准备 Gemma 4 容器

只有在 Qwen2.5 baseline 跑通之后，再开始 Gemma 4 容器路线。不要同时推进两个复杂环境。

---

## 14. 参考资料

- vLLM GPU installation docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM Gemma 4 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vLLM releases: https://github.com/vllm-project/vllm/releases
- Google Gemma 4 model overview: https://ai.google.dev/gemma/docs/core
- NVIDIA CUDA compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

---

## 15. 一句话总结

当前环境已经成功搭好：

```text
vLLM 0.6.1 + PyTorch cu121 + A100 + 离线 conda-pack
```

它适合跑稳定 baseline，例如：

```text
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Llama-3.1-8B-Instruct
Mistral-7B-Instruct-v0.3
```

它**不适合直接跑 Gemma 4**。Gemma 4 应单独用新 vLLM 容器路线，优先 E2B/E4B，不要直接上 31B / 26B-A4B。
