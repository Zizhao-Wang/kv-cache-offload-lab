# KV Cache Offloading Research 新人教程：Part II
# GitHub 仓库组织、vLLM 源码管理、多人协作与迁移
> 目标读者：已经完成 Part I 的同学，或者准备参与 vLLM / KV cache offloading 开发的新同学。  
Part I 解决的问题：在一台 GPU 机器上搭好 Driver、micromamba、uv、vLLM editable install、CUDA/nvcc、FlashInfer JIT、小模型 smoke test。  
Part II 解决的问题：代码应该放哪里、GitHub 仓库怎么组织、vLLM 源码怎么改、怎么提交、怎么协作、换机器时怎么复现环境。  
核心原则：**GitHub 放代码、脚本、配置、文档、manifest；机器工作区放模型、环境、cache、日志、runs；vLLM 上游源码用 fork 管理，不要直接复制进实验仓库。**
>

---

## 0. Part II 和 Part I 的关系
Part I 的目标是把一台机器上的开发环境跑通：

```latex
NVIDIA Driver
  ↓
micromamba / vllm-dev
  ↓
uv
  ↓
vLLM editable install
  ↓
CUDA/nvcc in micromamba env
  ↓
FlashInfer / CUDA headers / JIT path
  ↓
Qwen / Gemma smoke test
```

Part II 的目标是把这套过程变成可以协作、可以迁移、可以复现的工程结构：

```latex
机器工作区：
  保存真实模型、环境、cache、日志、实验输出

自己的 GitHub lab 仓库：
  保存脚本、配置、文档、manifest、benchmark、patch、实验控制代码

vLLM fork 仓库：
  保存对 vLLM 源码本身的修改
```

不要把这三类东西混成一个目录。

---

## 1. 三个层次：机器工作区、lab 仓库、vLLM fork
推荐的本地结构是：

```latex
$LAB_ROOT/
├── envs/                         # micromamba 环境，不进 GitHub
├── models/                       # 模型权重，不进 GitHub
├── hf_cache/                     # Hugging Face cache，不进 GitHub
├── vllm_cache/                   # vLLM cache，不进 GitHub
├── torch_cache/                  # Torch cache，不进 GitHub
├── logs/                         # 日志，不进 GitHub，除非是小型示例日志
├── runs/                         # 原始实验结果，不进 GitHub
├── tmp/                          # 临时文件，不进 GitHub
└── repos/
    ├── kv-cache-offload-lab/     # 自己的 GitHub lab 仓库
    └── vllm/                     # vLLM 源码仓库，建议来自你的 vLLM fork
```

例如在 4090 本机上：

```bash
export LAB_ROOT=$HOME/vllm_lab
```

以后新人只要记住一句话：

```latex
$LAB_ROOT 是机器工作区。
$LAB_ROOT/repos/kv-cache-offload-lab 是自己的实验控制仓库。
$LAB_ROOT/repos/vllm 是 vLLM 源码仓库。
```

---

## 2. GitHub lab 仓库到底放什么？
GitHub lab 仓库应该放“小而关键、可复现”的东西。

### 2.1 应该放进 lab 仓库的内容
```latex
脚本：
  环境激活脚本
  模型下载脚本
  smoke test 脚本
  benchmark 脚本
  profiling 脚本
  日志解析脚本
  画图脚本

配置：
  模型列表
  路径模板
  实验配置 yaml/json
  Slurm 脚本模板
  vLLM fork/branch/commit manifest
  platform manifest
  environment manifest

文档：
  README
  Part I / Part II 教程
  troubleshooting
  migration guide
  experiment notes
  design notes

代码：
  workload generator
  metrics collector
  monitor
  runner
  wrapper scripts
  不直接侵入 vLLM 的外部实验代码

小结果：
  summary csv
  聚合后的表格
  小型图表数据
  最终画图脚本
```

### 2.2 不应该放进 lab 仓库的内容
```latex
不要上传：
  模型权重 *.safetensors / *.bin / *.pt / *.pth / *.gguf
  micromamba / conda 环境目录
  conda-pack tar.gz
  wheel 大包
  Hugging Face cache
  vLLM cache
  Torch cache
  大型原始日志
  大型 profiling trace
  大型实验输出
  token / key / password
```

这些大文件应该放在：

```latex
本机 $LAB_ROOT/models
本机 $LAB_ROOT/envs
本机 $LAB_ROOT/logs
本机 $LAB_ROOT/runs
课题组共享盘 / NAS / 对象存储
单独 artifact 目录
```

GitHub 只记录它们的：

```latex
路径
版本
下载方式
checksum
用途
复现命令
```

---

## 3. 推荐的 lab 仓库结构
建议：

```latex
kv-cache-offload-lab/
├── README.md
├── .gitignore
├── docs/
│   ├── part1_env_setup_dev_cuda.md
│   ├── part2_github_collaboration_vllm_source.md
│   ├── troubleshooting.md
│   ├── migration.md
│   └── design_notes/
├── scripts/
│   ├── local_4090/
│   │   ├── activate_vllm_dev.sh
│   │   ├── verify_env.sh
│   │   ├── test_flashinfer_sampler_qwen_small.py
│   │   ├── run_gemma4_local.py
│   │   └── run_qwen36_local_tp2.py
│   ├── models/
│   │   ├── download_qwen36_and_gemma4.sh
│   │   └── verify_model_downloads.sh
│   ├── slurm/
│   │   ├── smoke_test.slurm
│   │   └── benchmark_template.slurm
│   └── analysis/
│       ├── parse_vllm_logs.py
│       └── plot_latency.py
├── configs/
│   ├── paths.example.yaml
│   ├── models.yaml
│   └── experiments/
│       ├── smoke_qwen3_1_7b.yaml
│       ├── gemma4_e4b_local.yaml
│       └── qwen36_27b_tp2.yaml
├── manifests/
│   ├── platform.local_4090.yaml
│   ├── environment.vllm_dev.yaml
│   ├── models.local.yaml
│   └── vllm.current.yaml
├── src/
│   └── kv_offload_lab/
│       ├── __init__.py
│       ├── workload.py
│       ├── metrics.py
│       ├── monitor.py
│       └── runners.py
├── patches/
│   └── vllm/
│       └── README.md
└── results/
    ├── README.md
    └── summary/
```

其中：

```latex
scripts/local_4090/
  放 4090 本机相关脚本。

scripts/slurm/
  放 Slurm/HPC 任务脚本。不要和单机脚本混在一起。

manifests/
  记录当前机器、环境、模型、vLLM fork commit。

patches/
  只放小 patch 或 patch 说明。长期开发还是应进入 vLLM fork。

results/summary/
  只放小型汇总结果，不放原始大日志。
```

---

## 4. vLLM 源码到底怎么管理？
这是 Part II 最重要的问题。

### 4.1 不推荐：把 vLLM 源码复制进 lab 仓库并删除 `.git`
你提到的方案是：

```latex
把 vLLM 源码复制到自己的 GitHub 仓库里
删掉 vLLM 里面的 .git / .gitignore
把它当成 lab 仓库的一部分
```

这个方案**不推荐**。

原因：

```latex
1. 会丢失 vLLM 上游 Git 历史。
2. 很难同步 vLLM upstream 更新。
3. 很难看清你到底改了 vLLM 的哪些文件。
4. lab 仓库会变得很大、很乱。
5. 多人协作时容易误提交大量无关 vLLM 文件。
6. 以后想给 vLLM upstream 提 PR 会非常困难。
```

删除 `.git` 看起来简单，但其实是把一个独立项目“拍扁”成普通文件夹。短期省事，长期很痛苦。

---

### 4.2 推荐：自己的 lab 仓库 + vLLM fork 两个仓库
推荐结构：

```latex
GitHub:
  your-org/kv-cache-offload-lab
  your-org/vllm 或 your-user/vllm fork

本地:
  $LAB_ROOT/repos/kv-cache-offload-lab
  $LAB_ROOT/repos/vllm
```

其中：

```latex
kv-cache-offload-lab：
  管实验、脚本、文档、manifest、benchmark。

vllm fork：
  管 vLLM 源码修改。
```

这和我们当前本地结构最匹配：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab
$LAB_ROOT/repos/vllm
```

---

### 4.3 vLLM fork 的正确工作流
第一步，在 GitHub 上 fork：

```latex
vllm-project/vllm
  ↓ fork
your-user/vllm
```

第二步，本地设置 remote：

```bash
cd "$LAB_ROOT/repos/vllm"

git remote -v
```

如果当前 `origin` 是官方 vLLM，可以改名为 `upstream`：

```bash
git remote rename origin upstream
```

然后添加自己的 fork：

```bash
git remote add origin git@github.com:<your-user-or-org>/vllm.git
```

检查：

```bash
git remote -v
```

期望类似：

```latex
origin    git@github.com:<your-user-or-org>/vllm.git (fetch)
origin    git@github.com:<your-user-or-org>/vllm.git (push)
upstream  https://github.com/vllm-project/vllm.git (fetch)
upstream  https://github.com/vllm-project/vllm.git (push)
```

第三步，创建开发分支：

```bash
git switch -c kv-offload-dev
git push -u origin kv-offload-dev
```

以后所有 vLLM 源码修改都提交到这个分支。

---

## 5. lab 仓库如何记录当前使用的 vLLM 版本？
自己的 lab 仓库不要复制 vLLM 源码，但要记录使用了哪个 vLLM fork、哪个 branch、哪个 commit。

创建：

```latex
manifests/vllm.current.yaml
```

示例：

```yaml
vllm:
  upstream_repo: "https://github.com/vllm-project/vllm.git"
  fork_repo: "git@github.com:<your-user-or-org>/vllm.git"
  local_path: "$LAB_ROOT/repos/vllm"
  branch: "kv-offload-dev"
  commit: "<fill-with-git-rev-parse-HEAD>"
  install_method: "editable"
  install_command: "VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto"
  torch:
    version: "2.11.0+cu130"
    cuda: "13.0"
  cuda_toolkit:
    installed_in: "$LAB_ROOT/envs/vllm-dev"
    nvcc: "$LAB_ROOT/envs/vllm-dev/bin/nvcc"
    cuda_home: "$LAB_ROOT/envs/vllm-dev"
  notes:
    - "vLLM source is kept as a separate fork, not copied into lab repo."
    - "Lab repo stores scripts, manifests, docs, configs, patches, and benchmark code."
```

自动生成当前 commit：

```bash
cd "$LAB_ROOT/repos/vllm"
git rev-parse HEAD
```

---

## 6. 什么时候用 patch？
patch 适合：

```latex
临时记录某个小修改
给别人快速复现一个差异
在正式 fork 工作流没整理好之前做过渡
```

生成 patch：

```bash
cd "$LAB_ROOT/repos/vllm"
git diff > "$LAB_ROOT/repos/kv-cache-offload-lab/patches/vllm/my_change.patch"
```

应用 patch：

```bash
cd "$LAB_ROOT/repos/vllm"
git apply "$LAB_ROOT/repos/kv-cache-offload-lab/patches/vllm/my_change.patch"
```

但注意：

```latex
patch 不是长期主线。
长期改 vLLM 应该用 vLLM fork 分支。
```

推荐规则：

```latex
小实验/临时修改：
  可以保存 patch。

正式功能/长期原型：
  提交到 vLLM fork 的 branch。

论文实验版本：
  lab 仓库 manifest 固定 vLLM fork commit。
```

---

## 7. 是否可以用 git submodule？
可以，但不建议第一天就用。

submodule 结构类似：

```latex
kv-cache-offload-lab/
└── external/
    └── vllm/    # submodule 指向你的 vLLM fork
```

优点：

```latex
lab 仓库能记录精确 vLLM commit。
clone 时能把 vLLM 放在 lab 仓库下面。
```

缺点：

```latex
新人容易忘记 git submodule update --init --recursive。
分支切换和提交更容易混乱。
CI / 脚本要额外处理 submodule。
```

当前推荐先不用 submodule，而是用：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab
$LAB_ROOT/repos/vllm
manifests/vllm.current.yaml 记录 vLLM fork commit
```

等流程稳定后再考虑 submodule。

---

## 8. 是否可以用 git subtree 或 vendor copy？
可以，但只适合少数情况。

### 8.1 vendor copy
vendor copy 就是把 vLLM 源码复制进 lab 仓库。

不推荐作为主线。  
只有在下面情况才考虑：

```latex
需要做一个完全冻结的 artifact snapshot
不打算同步 upstream
不打算给 upstream 提 PR
只为了归档某个论文版本
```

即便这样，也应该放到类似：

```latex
third_party/vllm_snapshot/
```

并明确说明：

```latex
This is a frozen snapshot for artifact archival only.
Do not develop here.
```

### 8.2 git subtree
git subtree 比直接复制好一点，可以保留一部分同步能力，但使用复杂度高。  
当前阶段不建议。

---

## 9. 自己的 lab 仓库怎么初始化？
如果还没有 lab 仓库：

```bash
export LAB_ROOT=$HOME/vllm_lab

cd "$LAB_ROOT/repos"
mkdir -p kv-cache-offload-lab
cd kv-cache-offload-lab

git init
mkdir -p docs scripts/local_4090 scripts/models scripts/slurm scripts/analysis configs/experiments manifests src/kv_offload_lab patches/vllm results/summary

touch README.md
touch src/kv_offload_lab/__init__.py
```

写 `.gitignore`：

```bash
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
venv/
.env
.env.local

# Secrets
*.token
secrets.yaml
*.pem

# Large model files
*.safetensors
*.bin
*.pt
*.pth
*.gguf
models/

# Caches
hf_cache/
vllm_cache/
torch_cache/
.cache/

# Environments and packages
envs/
*.tar.gz
*.tgz
*.zip
wheelhouse/
offline_pkgs/
images/
*.sif

# Logs and experiment outputs
logs/
runs/
tmp/
*.log
*.out
*.err
*.trace
*.jsonl

# Editors / OS
.vscode/
.idea/
.DS_Store
EOF
```

第一次提交：

```bash
git add README.md .gitignore docs scripts configs manifests src patches results
git commit -m "Initialize KV cache offload lab repo"
```

连接远程 GitHub：

```bash
git remote add origin git@github.com:<your-user-or-org>/kv-cache-offload-lab.git
git branch -M main
git push -u origin main
```

---

## 10. 当前 4090 机器应该放哪些脚本？
建议把已经跑通过的脚本放到：

```latex
scripts/local_4090/
```

至少包括：

```latex
activate_vllm_dev.sh
verify_env.sh
test_flashinfer_sampler_qwen_small.py
run_gemma4_local.py
run_qwen36_local_tp2.py
```

### 10.1 `activate_vllm_dev.sh`
这个脚本负责：

```latex
激活 micromamba vllm-dev
设置 HF cache
设置 vLLM cache
设置 CUDA_HOME / CUDA_PATH
设置 nvcc 路径
设置 FlashInfer JIT 需要的 NVIDIA PyPI CUDA headers/libs 路径
```

示例路径：

```latex
$LAB_ROOT/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh
```

### 10.2 `verify_env.sh`
用于快速检查：

```latex
nvidia-smi
python
torch
vLLM
nvcc
CUDA_HOME
FlashInfer
模型目录
```

### 10.3 smoke test 脚本
不要总用：

```bash
python - <<'PY'
...
PY
```

尤其是 vLLM 触发 multiprocessing spawn 时，stdin 脚本可能报：

```latex
FileNotFoundError: ... <stdin>
```

所以正式脚本应该写成真实 `.py` 文件，并带：

```python
if __name__ == "__main__":
    main()
```

---

## 11. 多人协作分支策略
推荐简单规则：

```latex
main:
  稳定文档、脚本、manifest。

dev:
  当前集成开发分支。

feature/<name>:
  单个功能或实验分支。

exp/<date>-<topic>:
  临时实验分支。
```

例子：

```bash
git switch -c feature/qwen36-smoke-test
git add scripts/local_4090/run_qwen36_local_tp2.py manifests/models.local.yaml
git commit -m "Add Qwen3.6 local smoke test script"
git push -u origin feature/qwen36-smoke-test
```

合并前检查：

```bash
git status
git diff --stat main...
```

不要把下面这些意外提交：

```latex
models/
envs/
logs/
runs/
hf_cache/
*.safetensors
*.bin
token
```

---

## 12. vLLM fork 的分支策略
vLLM fork 推荐：

```latex
main:
  不直接开发，用于跟 upstream 对齐。

kv-offload-dev:
  当前主开发分支。

kv-offload-prototype/<feature>:
  某个具体原型，例如 block eviction、CPU offload、NVMe tier。

debug/<issue>:
  临时调试分支。
```

同步 upstream：

```bash
cd "$LAB_ROOT/repos/vllm"

git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

更新开发分支：

```bash
git switch kv-offload-dev
git merge main
git push origin kv-offload-dev
```

如果冲突复杂，可以改用 rebase，但新人先用 merge 更直观。

---

## 13. lab 仓库和 vLLM fork 如何对应？
lab 仓库记录实验和环境，vLLM fork 记录源码修改。  
两者通过 manifest 连接。

例如：

```latex
kv-cache-offload-lab/manifests/vllm.current.yaml
```

记录：

```yaml
vllm:
  fork_repo: "git@github.com:<your-user-or-org>/vllm.git"
  branch: "kv-offload-dev"
  commit: "abc123..."
experiment:
  script: "scripts/local_4090/run_qwen36_local_tp2.py"
  model: "Qwen3.6-27B"
  platform: "local_4090"
```

这样别人复现时知道：

```latex
应该 clone 哪个 vLLM fork
checkout 哪个 commit
用哪个脚本跑
模型应该放在哪里
```

---

## 14. 环境、模型、平台 manifest
### 14.1 环境 manifest
```latex
manifests/environment.vllm_dev_local_4090.yaml
```

示例：

```yaml
name: vllm-dev
machine: local_4090
python: "3.12.13"
env_manager: "micromamba"
env_path: "$LAB_ROOT/envs/vllm-dev"
package_manager: "uv"
vllm:
  install_method: "editable"
  version: "0.20.1rc1.dev91+ga749a33d8.precompiled"
torch:
  version: "2.11.0+cu130"
  cuda: "13.0"
cuda:
  driver: "595.58.03"
  nvidia_smi_cuda: "13.2"
  nvcc: "13.0"
  cuda_home: "$LAB_ROOT/envs/vllm-dev"
flashinfer:
  status: "installed and JIT path configured"
notes:
  - "CUDA/nvcc installed inside micromamba env, not system-wide."
  - "PyPI NVIDIA cu13 headers may need CPATH for curand.h."
```

### 14.2 模型 manifest
```latex
manifests/models.local_4090.yaml
```

示例：

```yaml
models:
  - name: Qwen3.6-27B
    hf_repo: "Qwen/Qwen3.6-27B"
    local_dir: "$LAB_ROOT/models/Qwen3.6-27B"
    files: 29
    safetensor_shards: 15
    intended_use: "large local model test, TP=2 first"

  - name: gemma-4-E4B-it
    hf_repo: "google/gemma-4-E4B-it"
    local_dir: "$LAB_ROOT/models/gemma-4-E4B-it"
    files: 9
    intended_use: "small/medium local model smoke test"
```

### 14.3 平台 manifest
```latex
manifests/platform.local_4090.yaml
```

示例：

```yaml
platform:
  name: "local_4090"
  os: "Ubuntu 24.04"
  gpu:
    count: 2
    model: "NVIDIA GeForce RTX 4090"
    memory_per_gpu: "24GB"
  driver: "595.58.03"
  nvidia_smi_cuda: "13.2"
  workdir: "$LAB_ROOT"
  storage:
    root_fs: "EXT4"
    notes:
      - "Models and cache are currently under $HOME/vllm_lab."
      - "For heavier experiments, prefer a dedicated local NVMe/data disk if available."
```

---

## 15. 迁移到新机器的流程
不要幻想：

```latex
git clone 之后一切都有。
```

GitHub 只保存代码和 manifest。  
大文件和环境要重新安装或单独搬。

### 15.1 新机器检查
```bash
hostname
cat /etc/os-release
uname -a
nvidia-smi
df -hT
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE
```

判断：

```latex
GPU 是否满足模型需求？
Driver 是否足够新？
是否有本地 NVMe？
是否能联网？
OS 是否适合当前 vLLM / PyTorch？
```

### 15.2 建工作区
```bash
export LAB_ROOT=$HOME/vllm_lab
mkdir -p "$LAB_ROOT"/{envs,repos,models,wheelhouse,offline_pkgs,images,runs,logs,tmp,hf_cache,vllm_cache,torch_cache,kv_cache_offload}
```

### 15.3 clone lab 仓库
```bash
cd "$LAB_ROOT/repos"
git clone git@github.com:<your-user-or-org>/kv-cache-offload-lab.git
```

### 15.4 clone vLLM fork
```bash
cd "$LAB_ROOT/repos"
git clone git@github.com:<your-user-or-org>/vllm.git
cd vllm
git checkout <manifest-recorded-branch-or-commit>
```

### 15.5 按 Part I 重建环境
```latex
安装 driver
安装 micromamba
创建 vllm-dev
安装 uv
editable 安装 vLLM
安装 CUDA/nvcc 到 vllm-dev
配置 activation 脚本
下载或复制模型
跑 smoke test
```

---

## 16. 当前研究路线建议
### 阶段 1：稳定运行
目标：

```latex
Qwen 小模型 smoke test
Gemma 4 E4B 本地测试
Qwen3.6-27B 本地 TP=2 测试
```

做：

```latex
记录 TTFT / TPOT / output toks/s
记录 GPU memory
记录 KV cache tokens
记录 max_model_len / gpu_memory_utilization
```

### 阶段 2：理解 vLLM KV cache 行为
做：

```latex
读 scheduler / block manager / worker / cache engine
加日志
记录 block allocation
记录 GPU KV cache size
记录 request 生命周期
```

### 阶段 3：做 offloading 原型
先从 Python / 系统逻辑层做起：

```latex
哪些 block 是 hot？
哪些 block 是 cold？
什么时候触发显存压力？
CPU DRAM 和 NVMe 访问成本是多少？
```

### 阶段 4：深入修改 vLLM
根据需要修改：

```latex
cache engine
block manager
scheduler
worker
attention backend interface
```

必要时再进入：

```latex
CUDA / C++ / Triton kernel
```

---

## 17. 一句话总结
新人先记住：

```latex
1. $LAB_ROOT 是机器工作区，不是 GitHub 仓库。
2. kv-cache-offload-lab 是自己的实验控制仓库，放脚本、文档、配置、manifest。
3. vLLM 源码应该用 fork 单独管理，不要复制进 lab 仓库后删除 .git。
4. 模型、环境、cache、logs、runs 不进 GitHub。
5. 复现靠 manifest + 脚本 + vLLM fork commit，而不是把所有东西塞进一个仓库。
```

