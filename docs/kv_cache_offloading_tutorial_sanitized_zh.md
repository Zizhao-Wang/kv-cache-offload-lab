# KV Cache Offloading Research 新人教程：推理环境、平台流程、GitHub 协同与迁移

> 目标读者：刚开始做 LLM 推理 / vLLM / KV cache offloading 的同学。  
> 目标：看完后知道“要下载什么、放在哪里、怎么跑、在哪里改代码、什么该进 GitHub、什么不该进 GitHub、以后怎么迁移到新机器”。  
> 说明：本文使用占位符和环境变量，不包含个人用户名。请根据实际平台替换路径。

---

## 0. 先讲清楚：这份教程解决什么问题

做 KV cache offloading research 时，容易把几类东西混在一起：

1. **模型权重**：例如 Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct。它们是很大的文件，一般几十 GB。
2. **推理框架 / 引擎**：例如 vLLM。它负责加载模型、管理 KV cache、调度请求、调用 GPU kernel。
3. **Python 环境**：例如 Python、PyTorch、Transformers、vLLM 依赖包。
4. **CUDA / Driver / Kernel**：底层 GPU 运行环境。
5. **实验代码**：你的 benchmark、profiling、offloading 策略、日志解析脚本。
6. **实验数据与日志**：每次运行产生的结果、throughput/latency、GPU/CPU/SSD 监控日志。
7. **平台文件系统**：本地 NVMe、NFS、并行文件系统、home 目录等。

这几类东西**不能随便混放**。尤其是模型、KV cache offloading 文件、实验输出，应该放在计算节点的本地高速盘上，而不是网络文件系统。

---

# Part I：给新人看的 LLM 推理 / vLLM / CUDA / Hugging Face / KV cache offloading 基础和平台流程

## 1. LLM 推理最小链路

一次最普通的 LLM 推理大概是这样：

```text
用户 prompt
  ↓
tokenizer 把文本变成 token ids
  ↓
vLLM / 推理引擎加载模型权重
  ↓
prefill 阶段：处理输入 prompt，生成初始 KV cache
  ↓
decode 阶段：逐 token 生成输出，同时持续读取/追加 KV cache
  ↓
tokenizer 把输出 token ids 转回文本
```

对 KV cache offloading 来说，最重要的是这条路径：

```text
GPU HBM 显存
  ↔ CPU DRAM 内存
  ↔ 本地 NVMe / SSD
```

不要把实验热路径放到：

```text
GPU ↔ CPU ↔ 网络文件系统 / 分布式文件系统
```

因为这样测到的可能是网络存储的延迟和抖动，而不是你设计的 KV cache offloading 策略。

---

## 2. 常见组件解释

### 2.1 模型权重是什么

模型权重就是 LLM 的参数文件，例如：

```text
model.safetensors
model-00001-of-00004.safetensors
config.json
tokenizer.json
tokenizer_config.json
```

模型权重一般来自 Hugging Face。它们很大，不应该上传到 GitHub。

### 2.2 Tokenizer 是什么

Tokenizer 负责把文本和 token ids 互相转换。很多 Instruct 模型还有 chat template，例如：

```text
system message
user message
assistant prefix
```

如果 tokenizer 配错，模型可能能运行，但输出格式会不对。

### 2.3 vLLM 是什么

vLLM 是一个高性能 LLM 推理框架。它常用于：

- 高吞吐推理；
- OpenAI API 兼容服务；
- continuous batching；
- KV cache block 管理；
- 多 GPU tensor parallel；
- prefix cache / paged attention 等推理优化。

做 KV cache offloading research 时，vLLM 是一个合适的研究基线，因为它已经有较成熟的 KV cache 管理结构。

### 2.4 Hugging Face 是什么

Hugging Face 是模型仓库。我们通常从那里下载：

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
meta-llama/...
mistralai/...
```

下载命令通常是：

```bash
hf download <repo-id> --local-dir <local-model-dir>
```

不同版本的 Hugging Face CLI 参数不完全一样。某些旧教程里的 `--local-dir-use-symlinks False` 在一些新版 `hf download` 中已经不存在，因此不要盲目复制旧命令。

### 2.5 CUDA、Driver、PyTorch CUDA runtime 是什么关系

容易混淆的点：

```text
NVIDIA Driver：系统级 GPU 驱动，由管理员维护。
CUDA Toolkit：编译 CUDA 程序用的工具链，例如 nvcc、headers、libs。
PyTorch CUDA runtime：PyTorch wheel 自带或依赖的一组 CUDA 用户态库。
nvidia-smi 里的 CUDA Version：表示当前 driver 支持的 CUDA 运行能力上限，不等于系统必须安装那个版本的 CUDA Toolkit。
```

所以看到：

```text
nvidia-smi 显示 CUDA Version 12.2
```

不代表必须在系统里安装 CUDA 12.2，也不代表不能运行 PyTorch cu121 wheel。只要 driver 与 wheel 所需 CUDA runtime 兼容，PyTorch 就可以识别 GPU。

### 2.6 GPU kernel 是什么

这里的 kernel 不是 Linux kernel，而是 GPU 上执行的小程序，例如 attention kernel、matmul kernel、copy kernel。vLLM 里有很多 C++/CUDA 或 Triton kernel。

如果你只是改 Python 层调度、日志、benchmark，通常不需要重新编译 kernel。  
如果你要改 attention 内核、KV cache layout 的底层 GPU 访问方式，就可能需要编译 C++/CUDA/Triton 相关部分，难度和环境风险都会明显上升。

### 2.7 KV cache 是什么

Transformer decode 阶段每生成一个 token，都需要历史 token 的 Key/Value。为了不重复计算历史 token，系统会缓存这些 Key/Value，这就是 KV cache。

KV cache 的特点：

```text
上下文越长，KV cache 越大。
batch 越大，KV cache 越大。
模型层数越多、hidden size 越大，KV cache 越大。
```

当 GPU 显存不够时，就会产生 KV cache offloading 的研究问题：

```text
哪些 KV 留在 GPU？
哪些 KV 放到 CPU DRAM？
哪些 KV 放到本地 NVMe？
什么时候搬？
怎么避免搬运阻塞 decode？
怎么降低重载延迟？
```

---

## 3. 平台上为什么不能“随便选一个文件夹”

模型和代码从语义上当然可以放在任意目录，但从实验角度不能随便放。

原因如下：

1. **性能问题**：模型加载、KV offloading、日志写入如果走 NFS/并行文件系统，会引入网络延迟和抖动。
2. **可复现问题**：同样代码放在不同文件系统，性能可能不一样。
3. **安全/权限问题**：计算节点通常没有公网，不能直接下载包或模型。
4. **迁移问题**：如果路径写死到某个用户名或某个机器，换机器后全部失效。

因此不要说“模型必须放在某个固定路径”，而应该说：

> 模型应放在**计算节点本地高速盘**上的项目工作区中。具体路径由平台决定，用环境变量统一配置。

推荐抽象成：

```bash
export LOCAL_WORKDIR=/path/to/local_nvme/$USER/vllm_lab
```

在本平台上，如果本地 NVMe 挂载点是 `/ssdcache`，则可以设为：

```bash
export LOCAL_WORKDIR=/ssdcache/$USER/vllm_lab
```

如果换到新机器，本地盘可能叫：

```text
/local/$USER/vllm_lab
/scratch/$USER/vllm_lab
/nvme/$USER/vllm_lab
/data/local/$USER/vllm_lab
```

关键不是路径名字，而是确认它是不是本地盘：

```bash
findmnt -T "$LOCAL_WORKDIR"
df -hT "$LOCAL_WORKDIR"
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE
```

应该看到本地 ext4/xfs/NVMe，而不是 nfs、lustre、gpfs、parastor、ceph 等网络/并行文件系统。

---

## 4. 推荐工作区结构

统一使用一个根目录：

```bash
export LOCAL_WORKDIR=/path/to/local_nvme/$USER/vllm_lab
```

目录结构：

```text
$LOCAL_WORKDIR/
├── envs/              # Python/conda-pack 环境
├── repos/             # vLLM 源码、自己的实验源码
├── models/            # 模型权重，本地保存，不进 GitHub
├── wheelhouse/        # 离线 wheel 包
├── offline_pkgs/      # 从 admin 传来的环境包、源码包、校验文件
├── images/            # 容器镜像，例如 .sif
├── runs/              # 每次实验的结果目录
├── logs/              # Slurm 日志、服务日志
├── tmp/               # 临时文件
├── hf_cache/          # Hugging Face cache
├── vllm_cache/        # vLLM cache
├── torch_cache/       # Torch cache
└── kv_cache_offload/  # KV cache offloading 文件
```

创建命令：

```bash
mkdir -p "$LOCAL_WORKDIR"/{envs,repos,models,wheelhouse,offline_pkgs,images,runs,logs,tmp,hf_cache,vllm_cache,torch_cache,kv_cache_offload}
```

---

## 5. 管理节点与计算节点的分工

很多集群是这样的：

```text
admin / login 节点：
  有公网。
  负责下载 GitHub repo、Python wheel、模型权重。
  不跑正式 GPU 实验。

compute 节点：
  有 GPU 和本地 NVMe。
  没公网，或者不允许接公网。
  只做离线解包、推理、实验。
```

正确流程：

```text
admin 下载 / 构建 / 打包
  ↓
rsync/scp 到 compute 的本地 NVMe
  ↓
compute 离线运行
```

不要在 compute 节点上执行：

```bash
yum install ...
pip install ...
git clone https://...
hf download ...
wget ...
curl ...
```

---

## 6. 一个稳定 baseline 环境应该怎么准备

推荐用两类环境：

```text
stable native environment：
  用于跑当前可用 baseline，版本固定，尽量不动。

future/container environment：
  用于新模型、新 vLLM、新机器迁移。
```

### 6.1 native 环境的思路

在 admin 节点：

1. 用 micromamba/conda 创建 Python 3.10 环境；
2. 用 pip 安装指定版本 vLLM；
3. 解决依赖坑；
4. 用 conda-pack 打包；
5. rsync 到 compute；
6. compute 上解包、运行 `conda-unpack`、离线使用。

### 6.2 为什么不要随便升级 CUDA / Driver

Driver 是系统级组件，通常由管理员维护。随便升级可能导致：

```text
影响其他用户；
需要重启节点；
破坏已有 CUDA/PyTorch 环境；
和集群管理策略冲突。
```

研究环境应该尽量隔离在用户目录中，不修改系统级组件。

---

## 7. 当前已知的常见坑与处理方式

### 7.1 compute 节点没公网

现象：

```bash
ping -c 3 8.8.8.8
```

输出：

```text
connect: Network is unreachable
```

处理：不要修网络，不要加代理，不要给计算节点接公网。去 admin 下载，再传到 compute。

### 7.2 `hf download` 参数不兼容

某些环境中不支持：

```bash
--local-dir-use-symlinks False
```

应改为：

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
```

### 7.3 vLLM 0.6.x 依赖 `outlines -> pyairports`

如果使用：

```bash
python -m pip install --only-binary=:all: "vllm==0.6.1.post1"
```

可能遇到：

```text
outlines depends on pyairports
No matching distributions available for pyairports
```

解决方法是在 admin 上先构建 `pyairports` wheel：

```bash
python -m pip wheel --no-deps --wheel-dir ./wheelhouse "pyairports==0.0.1"

python -m pip install \
  --find-links ./wheelhouse \
  --only-binary=:all: \
  "vllm==0.6.1.post1"
```

### 7.4 `conda-pack` 报 pip 被覆盖

不要在 conda/micromamba 环境里执行：

```bash
python -m pip install --upgrade pip setuptools wheel
```

这可能导致 conda-pack 发现 pip 文件被 pip 自己覆盖。正确方式是：

```text
conda/micromamba 管 python、pip、setuptools、wheel、conda-pack；
pip 只安装 vLLM、torch、transformers 等普通 Python 包。
```

### 7.5 `conda-unpack` 用了系统 Python

如果 compute 上直接执行：

```bash
./bin/conda-unpack
```

报 Python 语法错误，可能是它调用了系统 Python 2。应使用环境自己的 Python：

```bash
./bin/python ./bin/conda-unpack
```

---

## 8. 模型下载流程：以 Qwen2.5-7B/14B 为例

### 8.1 admin 节点下载

```bash
mkdir -p ~/offline_models/qwen25
cd ~/offline_models/qwen25

hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./Qwen2.5-7B-Instruct \
  --max-workers 8

hf download Qwen/Qwen2.5-14B-Instruct \
  --local-dir ./Qwen2.5-14B-Instruct \
  --max-workers 8
```

建议后台运行：

```bash
mkdir -p logs
nohup hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./Qwen2.5-7B-Instruct \
  --max-workers 8 \
  > logs/qwen25_7b.log 2>&1 &

nohup hf download Qwen/Qwen2.5-14B-Instruct \
  --local-dir ./Qwen2.5-14B-Instruct \
  --max-workers 8 \
  > logs/qwen25_14b.log 2>&1 &
```

### 8.2 生成校验文件

```bash
find Qwen2.5-7B-Instruct -type f -print0 | sort -z | xargs -0 sha256sum > Qwen2.5-7B-Instruct.SHA256SUMS
find Qwen2.5-14B-Instruct -type f -print0 | sort -z | xargs -0 sha256sum > Qwen2.5-14B-Instruct.SHA256SUMS
```

### 8.3 传到 compute 节点

```bash
rsync -avP --partial --append-verify \
  ./Qwen2.5-7B-Instruct/ \
  <compute-node>:${LOCAL_WORKDIR}/models/Qwen2.5-7B-Instruct/

rsync -avP --partial --append-verify \
  ./Qwen2.5-14B-Instruct/ \
  <compute-node>:${LOCAL_WORKDIR}/models/Qwen2.5-14B-Instruct/
```

如果远程 shell 没有 `LOCAL_WORKDIR` 变量，就写成实际路径，例如：

```bash
rsync -avP ./Qwen2.5-7B-Instruct/ <compute-node>:/path/to/local_nvme/$USER/vllm_lab/models/Qwen2.5-7B-Instruct/
```

### 8.4 compute 节点检查

```bash
du -sh "$LOCAL_WORKDIR/models/Qwen2.5-7B-Instruct"
du -sh "$LOCAL_WORKDIR/models/Qwen2.5-14B-Instruct"
findmnt -T "$LOCAL_WORKDIR/models/Qwen2.5-7B-Instruct"
```

---

## 9. 最小 vLLM 推理测试

激活环境：

```bash
source "$LOCAL_WORKDIR/activate_vllm061.sh"
```

运行 Qwen2.5-7B：

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os

model_path = os.path.join(os.environ["LOCAL_WORKDIR"], "models/Qwen2.5-7B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用一段话解释什么是 KV cache offloading。"},
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
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

outputs = llm.generate(
    [prompt],
    SamplingParams(max_tokens=128, temperature=0.0),
)

print(outputs[0].outputs[0].text)
PY
```

第一阶段建议：

```text
先跑 7B；
7B 跑通后再跑 14B；
先用 max_model_len=2048/4096；
不要一上来跑超长上下文。
```

---

# Part II：GitHub 仓库组织、上传规则、环境迁移与多人协同

## 10. GitHub 仓库到底放什么

GitHub 应该放**小而关键、可复现的信息**，不应该放大文件。

### 10.1 应该上传 GitHub 的内容

```text
代码：
  benchmark 脚本
  profiling 脚本
  日志解析脚本
  offloading 策略代码
  vLLM patch / fork / wrapper

配置：
  实验配置 yaml/json
  模型列表
  workload 列表
  Slurm 脚本模板
  环境 manifest

文档：
  README
  setup guide
  troubleshooting
  experiment notes
  migration guide

小结果：
  summary csv
  聚合后的表格
  画图脚本
  小型图表数据
```

### 10.2 不应该上传 GitHub 的内容

```text
不要上传：
  模型权重 *.safetensors / *.bin
  conda-pack 环境 tar.gz
  wheel 大包
  Hugging Face cache
  vLLM cache
  大型原始日志
  大型 profiling trace
  大型实验输出
  token / key / password
```

这些应放在：

```text
本地 NVMe 工作区
admin 离线包目录
对象存储 / NAS / 课题组共享盘
release artifact 存储
```

GitHub 只记录它们在哪里、版本是什么、checksum 是什么。

---

## 11. 推荐 GitHub 仓库结构

建议创建一个自己的研究仓库，例如：

```text
kv-cache-offload-lab/
├── README.md
├── docs/
│   ├── 00_concepts.md
│   ├── 01_platform_setup.md
│   ├── 02_offline_install.md
│   ├── 03_model_download.md
│   ├── 04_migration.md
│   └── troubleshooting.md
├── configs/
│   ├── models.yaml
│   ├── paths.example.yaml
│   ├── experiments/
│   │   ├── smoke_qwen25_7b.yaml
│   │   └── long_context_qwen25_7b.yaml
├── scripts/
│   ├── admin/
│   │   ├── download_models.sh
│   │   ├── build_vllm_env.sh
│   │   └── pack_env.sh
│   ├── compute/
│   │   ├── activate_env.sh
│   │   ├── verify_env.sh
│   │   └── run_smoke_test.sh
│   ├── slurm/
│   │   ├── qwen25_7b_smoke.slurm
│   │   └── qwen25_14b_smoke.slurm
│   └── analysis/
│       ├── parse_vllm_logs.py
│       └── plot_latency.py
├── src/
│   ├── kv_offload_lab/
│   │   ├── __init__.py
│   │   ├── workload.py
│   │   ├── metrics.py
│   │   ├── monitor.py
│   │   └── runners.py
├── patches/
│   └── vllm061/
│       └── README.md
├── manifests/
│   ├── environment.vllm061.yaml
│   ├── models.qwen25.yaml
│   └── platform.current.yaml
├── results/
│   ├── README.md
│   └── summary/
├── .gitignore
└── pyproject.toml
```

---

## 12. vLLM 源码怎么管理

有三种方式：

### 12.1 方式 A：只用 pip wheel，不改 vLLM

适合初期跑 baseline：

```text
你的仓库只放 benchmark 和脚本。
vLLM 作为环境里的依赖存在。
不改 vLLM 源码。
```

优点：稳定、简单。  
缺点：不能实现深入 offloading 策略。

### 12.2 方式 B：fork vLLM，在 fork 里改

适合你要改 vLLM 内部：

```text
GitHub fork vLLM。
创建分支：kv-offload-prototype。
在 vLLM fork 里改 block manager / scheduler / worker / cache engine。
你的实验仓库记录 fork commit hash。
```

优点：适合真正改 vLLM。  
缺点：仓库大，和上游同步有成本。

### 12.3 方式 C：你的仓库保存 patch

适合研究早期：

```text
保持 vLLM 原始源码在本地。
你的仓库保存 patch 文件。
需要时 apply patch。
```

生成 patch：

```bash
cd $LOCAL_WORKDIR/repos/vllm
git diff > /path/to/kv-cache-offload-lab/patches/vllm061/my_change.patch
```

应用 patch：

```bash
cd $LOCAL_WORKDIR/repos/vllm
git apply /path/to/kv-cache-offload-lab/patches/vllm061/my_change.patch
```

推荐当前阶段用：

```text
自己的实验仓库 + vLLM 源码副本 + patch 记录
```

等策略稳定后再 fork vLLM。

---

## 13. 环境、模型、实验结果应该怎么记录

### 13.1 环境 manifest

创建：

```text
manifests/environment.vllm061.yaml
```

内容示例：

```yaml
name: vllm061-native
python: "3.10.20"
vllm: "0.6.1.post1"
torch: "2.4.0+cu121"
cuda_runtime: "12.1"
install_method: "admin-side micromamba + pip + conda-pack"
artifact:
  filename: "vllm061_env_py310.tar.gz"
  stored_at: "<admin-offline-artifact-dir>"
  deployed_to: "$LOCAL_WORKDIR/envs/vllm061-native"
notes:
  - "Do not pip upgrade pip inside conda env."
  - "Run conda-unpack via ./bin/python ./bin/conda-unpack if system Python is old."
```

### 13.2 模型 manifest

创建：

```text
manifests/models.qwen25.yaml
```

内容示例：

```yaml
models:
  - name: Qwen2.5-7B-Instruct
    hf_repo: Qwen/Qwen2.5-7B-Instruct
    local_dir: "$LOCAL_WORKDIR/models/Qwen2.5-7B-Instruct"
    checksum_file: "Qwen2.5-7B-Instruct.SHA256SUMS"
    intended_use: "baseline, smoke test, moderate context experiments"

  - name: Qwen2.5-14B-Instruct
    hf_repo: Qwen/Qwen2.5-14B-Instruct
    local_dir: "$LOCAL_WORKDIR/models/Qwen2.5-14B-Instruct"
    checksum_file: "Qwen2.5-14B-Instruct.SHA256SUMS"
    intended_use: "larger-memory-pressure experiments"
```

### 13.3 平台 manifest

创建：

```text
manifests/platform.current.yaml
```

内容示例：

```yaml
platform:
  node_role:
    admin: "downloads packages, repos, models"
    compute: "offline GPU inference"
  compute_node:
    os: "CentOS 7.9"
    gpu: "A100-PCIE-40GB"
    driver: "535.54.03"
    network_policy: "no internet on compute node"
    local_workdir: "$LOCAL_WORKDIR"
  storage_policy:
    use_for_hot_path:
      - "local NVMe"
    avoid_for_kv_offload:
      - "NFS"
      - "parallel/distributed filesystem"
      - "home directory if network-backed"
```

---

## 14. `.gitignore` 推荐

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# Large model files
*.safetensors
*.bin
*.pt
*.pth
*.gguf
models/
hf_cache/
vllm_cache/
torch_cache/

# Packed environments and wheels
*.tar.gz
*.tgz
*.zip
wheelhouse/
offline_pkgs/
images/
*.sif

# Logs and runs
logs/
runs/
*.log
*.out
*.trace
*.jsonl

# Secrets
*.token
.env.local
secrets.yaml
```

如果需要提交小型 summary，放到：

```text
results/summary/
```

不要提交原始大日志。

---

## 15. 新机器迁移流程

迁移时不要幻想“git clone 之后一切都有”。GitHub 只保存代码和 manifest，大文件要单独搬。

### 15.1 在新机器上检查

```bash
hostname
nvidia-smi
getconf GNU_LIBC_VERSION
df -hT
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE
which apptainer || which singularity || true
```

判断：

```text
GPU 是否满足模型需求？
Driver 是否能运行现有 PyTorch/vLLM？
是否有本地 NVMe？
是否有公网？如果没有，是否有 admin 节点？
OS/glibc 是否比旧机器更新？
```

### 15.2 设置新机器工作区

```bash
export LOCAL_WORKDIR=/path/to/local_nvme/$USER/vllm_lab
mkdir -p "$LOCAL_WORKDIR"/{envs,repos,models,wheelhouse,offline_pkgs,images,runs,logs,tmp,hf_cache,vllm_cache,torch_cache,kv_cache_offload}
```

### 15.3 拉 GitHub 仓库

```bash
cd "$LOCAL_WORKDIR/repos"
git clone <your-github-repo-url> kv-cache-offload-lab
```

### 15.4 复制离线环境和模型

从旧机器或 admin artifact 目录复制：

```text
vllm061_env_py310.tar.gz
Qwen2.5-7B-Instruct/
Qwen2.5-14B-Instruct/
SHA256SUMS
```

解包环境：

```bash
mkdir -p "$LOCAL_WORKDIR/envs/vllm061-native"
cd "$LOCAL_WORKDIR/envs/vllm061-native"
tar -xzf "$LOCAL_WORKDIR/offline_pkgs/vllm061_env_py310.tar.gz"
./bin/python ./bin/conda-unpack
```

### 15.5 验证

```bash
source "$LOCAL_WORKDIR/activate_vllm061.sh"
python - <<'PY'
import torch, vllm
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(vllm.__version__)
PY
```

### 15.6 跑 smoke test

先跑小模型 / 短上下文：

```text
Qwen2.5-7B-Instruct
max_model_len=2048 or 4096
max_tokens=128
batch size=1
```

通过后再扩大。

---

## 16. 当前研究路线建议

### 阶段 1：先跑起来

目标：稳定跑 Qwen2.5-7B/14B。

做：

```text
加载模型；
跑单 prompt；
跑 API server；
记录 TTFT、TPOT、throughput；
确认所有路径在本地 NVMe。
```

### 阶段 2：理解 vLLM 的 KV cache 行为

做：

```text
读 vLLM block manager / scheduler / worker 相关代码；
加日志；
记录 block allocation、block table、GPU memory usage；
不要一开始改 CUDA kernel。
```

### 阶段 3：做 offloading 原型

先从 Python-level 或系统层 tracing 做起：

```text
哪些 KV block 是 hot？
哪些是 cold？
什么时候发生显存压力？
CPU DRAM 与本地 NVMe 的开销各是多少？
```

### 阶段 4：深入改 vLLM

等理解足够后，再考虑：

```text
修改 cache engine；
修改 block manager；
修改 scheduler；
引入 CPU/NVMe tier；
必要时再碰 C++/CUDA/Triton kernel。
```

---

## 17. 一句话总结

新人先记住三件事：

```text
1. GitHub 放代码、配置、文档、manifest，不放模型和大环境。
2. admin 下载和打包，compute 离线运行。
3. 模型、cache、KV offload、实验输出放 compute 节点本地高速盘，不放网络文件系统。
```

对当前研究来说，最稳的起点是：

```text
vLLM 0.6.1 baseline + Qwen2.5-7B/14B + 本地 NVMe 工作区 + GitHub manifest 管理。
```
