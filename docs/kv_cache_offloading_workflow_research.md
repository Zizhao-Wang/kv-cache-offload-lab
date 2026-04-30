# KV Cache Offloading Research 新人教程：推理环境、平台流程、GitHub 协同与迁移

> 目标读者：刚开始做 LLM 推理 / vLLM / KV cache offloading 的同学。  
> 目标：看完后知道“从一台新机器开始，第一步要检查什么、安装什么、为什么这样装、后面哪些东西会影响 vLLM 和 KV cache offloading 开发”。  
> 说明：本文使用占位符和环境变量，不包含个人用户名。请根据实际平台替换路径。

---

# Part I：从一台新机器开始理解 LLM 推理环境

## 0. 为什么这份教程要从“安装流程”开始讲？

做 LLM 推理 / vLLM / KV cache offloading research 的时候，最容易让新人崩溃的不是某一条命令，而是这些概念全部混在一起：

```text
NVIDIA Driver
CUDA Toolkit
PyTorch CUDA wheel
vLLM
FlashAttention
Hugging Face 模型
KV cache
KV cache offloading
Docker / conda / micromamba
源码编译
```

如果一开始不把这些层次分清楚，后面报错时就很难判断：

```text
是 driver 没装好？
是 PyTorch 看不到 GPU？
是 CUDA Toolkit 版本不匹配？
是 vLLM 不支持这个模型？
是 FlashAttention 编译失败？
还是 KV cache offloading 逻辑本身有问题？
```

所以这份教程不先从“下载模型”开始，而是从一台新安装的 Linux 机器开始，按照实际搭建顺序解释每一层：

```text
新操作系统
  ↓
确认 GPU 硬件
  ↓
安装 NVIDIA Driver
  ↓
验证 nvidia-smi
  ↓
建立 Python / micromamba 环境
  ↓
安装 PyTorch
  ↓
验证 torch.cuda.is_available()
  ↓
安装 / 开发 vLLM
  ↓
下载 Hugging Face 模型
  ↓
运行推理
  ↓
修改 vLLM 源码做 KV cache offloading
```

---

## 1. LLM 推理最小链路：先知道每一步在干什么

在开始安装环境之前，先要知道我们最终要运行的东西是什么。一次最普通的 LLM 推理大概是这样：

```text
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

这条链路里，每一部分都有对应的系统组件和优化空间：

```text
文本输入 / 输出
  → tokenizer、chat template、prompt 格式

模型权重加载
  → Hugging Face 模型文件、safetensors、GPU/CPU 内存放置

prefill 阶段
  → 大量并行计算，主要压力在矩阵计算和 attention 计算

decode 阶段
  → 每次生成一个 token，反复读取历史 KV cache
  → 对延迟、调度、KV cache 管理非常敏感

KV cache 管理
  → 决定哪些 KV block 留在 GPU，哪些被释放、复用、换出或搬回

系统调度
  → vLLM 负责 batching、scheduler、block manager、worker、attention backend 等
```

所以，vLLM 不是简单地“调用模型生成文本”，而是在管理一整套推理系统：

```text
请求怎么排队？
batch 怎么合并？
KV cache 怎么分配？
显存不够时怎么办？
attention kernel 怎么读 KV？
多个请求之间怎么共享 prefix？
```

---

### 1.1 KV cache offloading 关注的是推理链路里的哪一段？

对 KV cache offloading 来说，最重要的是这条路径：

```text
GPU HBM 显存
  ↔ CPU DRAM 内存
  ↔ 本地 NVMe / SSD
```

为什么？

因为 decode 阶段会不断访问历史 KV cache。上下文越长、batch 越大、并发越高，KV cache 越容易变成显存瓶颈。

KV cache offloading 研究的核心不是简单地“把 KV 放到 SSD 上”，而是要回答：

```text
什么时候 GPU 显存不够？
哪些 KV cache 还值得留在 GPU？
哪些 KV cache 可以放到 CPU DRAM？
哪些 KV cache 可以进一步放到本地 NVMe？
什么时候搬出去？
什么时候搬回来？
搬回来会不会阻塞 decode？
搬运成本和重算成本哪个更划算？
```

---

### 1.2 为什么实验热路径不要放到网络文件系统？

做 KV cache offloading 时，热路径应该尽量使用本地高速盘，例如本地 NVMe / SSD。

不要把实验热路径放到：

```text
GPU ↔ CPU ↔ 网络文件系统 / 分布式文件系统
```

否则你测到的可能是网络存储的延迟、带宽竞争和系统抖动，而不是你设计的 KV cache offloading 策略本身。

更合理的实验路径是：

```text
GPU HBM
  ↔ CPU DRAM
  ↔ 本地 NVMe / SSD
```

网络文件系统更适合放：

```text
代码备份
小配置文件
最终结果归档
不在推理热路径上的数据
```

---

### 1.3 推理优化技术应该放在这条链路里理解

很多技术名字听起来很复杂，比如：

```text
FlashAttention
PagedAttention
continuous batching
prefix caching
speculative decoding
quantization
tensor parallel
KV cache offloading
```

不要把这些名字孤立地背下来。更好的理解方式是：它们分别优化推理链路里的不同位置。

例如：

```text
FlashAttention / 其他 attention kernel 优化
  → 主要优化 attention 计算和显存访问效率

PagedAttention / block-based KV cache 管理
  → 主要优化 KV cache 的组织、分配和复用

continuous batching
  → 主要优化多请求并发时的 GPU 利用率

prefix caching
  → 主要优化多个请求共享相同前缀时的重复计算

quantization
  → 主要减少模型权重和部分计算的显存/带宽压力

KV cache offloading
  → 主要解决长上下文/高并发下 KV cache 占用显存过大的问题
```

所以，FlashAttention 只是一个例子。它说明的是：

```text
推理系统会通过更高效的 attention 实现来减少计算和显存访问开销。
```

但它不是唯一技术，也不是所有系统都必须只依赖 FlashAttention。  
对我们来说，更重要的是学会把每项技术放回推理链路里，判断它到底优化了哪一段。

---


## 2. 第一步：确认系统能看到 NVIDIA GPU

在一台新机器上，第一步不是安装 vLLM，也不是安装模型，而是确认 Linux 能看到 GPU 硬件。

执行：

```bash
cat /etc/os-release
uname -a

lspci | grep -i nvidia || true
which nvidia-smi || true
nvidia-smi || true
```

如果 `lspci` 能看到类似：

```text
NVIDIA Corporation AD102 [GeForce RTX 4090]
```

说明硬件在 PCIe 层面已经被系统识别。

如果此时：

```text
nvidia-smi: command not found
```

这通常说明 NVIDIA driver 还没有装好，或者相关工具包还没安装。

这不是 vLLM 的问题，也不是 PyTorch 的问题，更不是模型的问题。  
这一步只说明：

```text
GPU 硬件存在，但系统还不能通过 NVIDIA driver 正常使用它。
```

---

## 3. 第二步：安装 NVIDIA Driver

### 3.1 NVIDIA Driver 是什么？

NVIDIA Driver 是 Linux 操作系统控制 GPU 的驱动。

它负责：

```text
1. 让 Linux 正确识别 NVIDIA GPU
2. 创建 /dev/nvidia* 设备文件
3. 允许 CUDA 程序向 GPU 提交任务
4. 提供 nvidia-smi
5. 让 PyTorch、vLLM、FlashAttention 等上层软件能使用 GPU
```

如果 driver 没有装好，后面所有东西都会失败：

```text
PyTorch 看不到 GPU
vLLM 无法使用 GPU
FlashAttention 无法运行
模型无法加载到 GPU
KV cache offloading 实验无法开始
```

所以第一步只做一件事：

```text
装好 NVIDIA Driver，并让 nvidia-smi 正常工作。
```

---

### 3.2 Ubuntu 上怎么选择 driver？

在 Ubuntu 上，不建议一开始手动去 NVIDIA 官网下载 `.run` 文件安装。  
更推荐使用 Ubuntu 自己的软件包系统安装 driver。

先安装检测工具：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
```

查看系统推荐的驱动：

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers list --gpgpu
```

输出里可能会看到很多版本，例如：

```text
nvidia-driver-535
nvidia-driver-580
nvidia-driver-595
nvidia-driver-595-open recommended
```

如果某个 driver 后面标了：

```text
recommended
```

通常优先选择这个版本。

例如系统推荐：

```text
nvidia-driver-595-open - recommended
```

那么可以安装：

```bash
sudo apt install -y nvidia-driver-595-open
sudo reboot
```

重启后验证：

```bash
nvidia-smi
```

期望看到 GPU，例如：

```text
NVIDIA GeForce RTX 4090
Driver Version: xxx
CUDA Version: xxx
```

---

## 4. nvidia-smi 里的 CUDA Version 到底是什么意思？

很多新人看到：

```text
nvidia-smi
CUDA Version: 12.x
```

就会以为：

```text
系统已经安装 CUDA Toolkit 12.x
```

这是不准确的。

`nvidia-smi` 里的 `CUDA Version` 更准确地说，是：

```text
当前 NVIDIA Driver 支持的 CUDA runtime 能力上限
```

它不等于：

```text
/usr/local/cuda 已经存在
nvcc 已经安装
系统已经安装 CUDA Toolkit
PyTorch 必须使用同样版本的 CUDA
```

所以：

```text
nvidia-smi 显示 CUDA Version 12.x
```

只说明 driver 够新，能够支持对应范围内的 CUDA 程序运行。

是否安装 CUDA Toolkit，是后面的开发问题，不是第一步必须做的事情。

---

## 5. CUDA 其实有三种含义

在机器学习环境里，“CUDA”这个词经常指不同的东西。

### 5.1 Driver 支持的 CUDA 能力

这就是 `nvidia-smi` 里显示的 CUDA Version。

它表示：

```text
当前 driver 能支持什么范围的 CUDA 程序运行
```

### 5.2 PyTorch wheel 自带或依赖的 CUDA runtime

当你安装 PyTorch 时，例如：

```bash
pip install torch
```

或者安装某个 CUDA 版本的 PyTorch wheel 时，PyTorch 包本身通常会带上或依赖一组 CUDA 用户态库，例如：

```text
CUDA runtime
cuDNN
cuBLAS
NCCL
其他 NVIDIA Python wheel 依赖
```

所以很多时候你不需要先安装系统级 CUDA Toolkit，也可以让 PyTorch 使用 GPU。

验证方式是：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果输出：

```text
cuda available: True
gpu: NVIDIA GeForce RTX 4090
```

说明 PyTorch 已经能使用 GPU。

### 5.3 CUDA Toolkit / nvcc

CUDA Toolkit 是开发工具包。

它里面最关键的是：

```text
nvcc
```

`nvcc` 是 CUDA 编译器。

只有当你要编译 CUDA 代码时，才真正需要 CUDA Toolkit，例如：

```text
改 vLLM 的 CUDA/C++ kernel
改 FlashAttention kernel
写自己的 .cu 文件
编译 torch custom op
完整从源码编译某些 GPU extension
```

因此第一天搭环境时，通常不要先急着执行：

```bash
sudo apt install cuda
sudo apt install cuda-toolkit-12-x
```

否则很容易让系统 CUDA、PyTorch CUDA wheel、环境变量 `LD_LIBRARY_PATH` 混在一起，后面排错更困难。

---

## 6. Driver、CUDA Toolkit、PyTorch 的正确安装顺序

推荐顺序是：

```text
1. 安装 NVIDIA Driver
2. 重启
3. 用 nvidia-smi 验证 GPU 正常
4. 创建 micromamba / Python 环境
5. 安装 PyTorch
6. 用 torch.cuda.is_available() 验证 PyTorch 能看到 GPU
7. 再安装或开发 vLLM
8. 只有需要编译 CUDA 代码时，再安装 CUDA Toolkit / nvcc
```

不推荐的顺序是：

```text
先装 CUDA Toolkit
再装 driver
再装 PyTorch
再装 vLLM
然后发现所有版本混在一起
```

一句话：

```text
Driver 是系统地基。
PyTorch CUDA wheel 是 Python 环境里的运行库。
CUDA Toolkit 是编译 CUDA 程序时才需要的工具箱。
```

---

## 7. Python 环境：为什么用 micromamba？

做 MLSys / vLLM 开发时，不建议直接污染系统 Python。

推荐使用：

```text
micromamba
```

原因：

```text
1. 环境隔离清楚
2. 删除环境方便
3. 复现实验方便
4. 不影响系统 Python
5. 后面可以记录 environment.yml 或 requirements.txt
```

推荐目录结构：

```text
$VLLM_LAB/
├── envs/              # Python / micromamba 环境
├── repos/             # vLLM 源码
├── models/            # Hugging Face 模型权重
├── hf_cache/          # Hugging Face cache
├── vllm_cache/        # vLLM cache
├── torch_cache/       # PyTorch cache
├── runs/              # 实验输出
├── logs/              # 日志
├── tmp/               # 临时文件
└── scripts/           # 启动脚本、benchmark 脚本
```

示例：

```bash
mkdir -p ~/vllm_lab/{envs,repos,models,hf_cache,vllm_cache,torch_cache,runs,logs,tmp,scripts}
```

---

## 8. PyTorch 和 vLLM 是什么关系？

PyTorch 是底层深度学习框架。

vLLM 是构建在 PyTorch / CUDA / 自定义 kernel 之上的大模型推理系统。

简单说：

```text
PyTorch 负责张量计算和 GPU 调用
vLLM 负责大模型推理系统逻辑
```

vLLM 会做：

```text
1. 加载 Hugging Face 模型
2. 管理请求队列
3. 做 continuous batching
4. 管理 KV cache block
5. 调用 attention backend
6. 调用 PyTorch / CUDA / Triton / FlashAttention
7. 提供 OpenAI-compatible server
```

对 KV cache offloading 来说，vLLM 是核心研究对象。

我们后面可能会改：

```text
scheduler
block manager
KV cache allocator
cache engine
prefix cache
worker
attention backend 接口
offload / swap / memory movement 路径
```

---

## 9. Attention 优化技术在这里是什么角色？

在 LLM 推理里，attention 是非常核心的计算部分。尤其在长上下文场景下，attention 不仅涉及大量计算，也涉及大量显存访问。

因此，很多推理系统都会使用 attention 优化技术，例如：

```text
FlashAttention
PagedAttention
Triton attention kernels
FlashInfer
xFormers attention
vendor-specific optimized kernels
vLLM 自己维护的 attention backend
```

这里不要把 FlashAttention 理解成唯一方案。它更适合作为一个例子，帮助新人理解：

```text
推理系统不仅要“把模型跑起来”，还要优化 attention 计算、显存访问和 KV cache 读取方式。
```

---

### 9.1 这些技术和 vLLM 的关系

vLLM 是推理系统。

attention kernel / attention backend 是 vLLM 在执行模型时调用的底层高性能组件。

可以这样理解：

```text
vLLM 负责系统逻辑：
  请求调度、batching、KV cache block 管理、prefix cache、worker 管理

attention backend 负责底层执行：
  如何高效计算 attention
  如何读取 Q/K/V
  如何访问 KV cache
  如何减少显存读写开销
```

所以它们不是同一层东西：

```text
vLLM 是系统框架
FlashAttention / PagedAttention / Triton kernel 等是可能被调用的底层优化技术
```

---

### 9.2 为什么 KV cache offloading 会和 attention backend 有关系？

decode 阶段每生成一个 token，都需要读取历史 KV cache。

简化来看：

```text
当前 token 的 Query 在 GPU 上
历史 Key/Value 存在 KV cache 里
attention 需要读取这些历史 Key/Value
```

如果所有 KV cache 都在 GPU HBM 里，attention backend 直接从 GPU 显存读。

但如果我们做 KV cache offloading，情况就复杂了：

```text
一部分 KV 在 GPU HBM
一部分 KV 在 CPU DRAM
一部分 KV 甚至在本地 NVMe
```

这时就会出现新的系统问题：

```text
attention kernel 真正计算前，KV 是否已经搬回 GPU？
搬运和计算能不能重叠？
哪些 KV 必须完整搬回？
是否可以只搬部分重要 KV？
搬运粒度是 token、block、page，还是 sequence？
```

因此，早期研究可以先改 Python 层的调度和 block 管理逻辑；但如果后面要深入优化性能，可能需要理解甚至修改底层 attention backend 或 CUDA/Triton kernel。

---

### 9.3 第一阶段不要急着改 attention kernel

刚开始做 KV cache offloading，不建议第一步就改 FlashAttention 或 CUDA kernel。

更推荐的顺序是：

```text
先理解 vLLM 的 request / scheduler / block manager
  ↓
再加 profiling，知道 KV cache 在哪里分配、什么时候释放
  ↓
再做 CPU DRAM / NVMe offload 的原型
  ↓
最后再判断是否需要修改 attention backend 或 kernel
```

原因是：

```text
attention kernel 层非常底层
编译和调试成本高
错误可能表现成 silent wrong output 或性能异常
很难区分是策略问题、数据搬运问题，还是 kernel 问题
```

所以对新人来说，先把推理链路和 vLLM 系统逻辑搞清楚，比一开始研究 FlashAttention 源码更重要。

---


## 10. Hugging Face 模型放在哪里？

模型权重一般来自 Hugging Face，例如：

```text
Qwen/Qwen3.6-xxx
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
meta-llama/...
mistralai/...
```

模型权重通常很大，不应该放进 GitHub。

推荐放在：

```text
$VLLM_LAB/models/
```

下载方式一般是：

```bash
hf download <repo-id> --local-dir $VLLM_LAB/models/<model-name>
```

注意：

```text
不同版本的 Hugging Face CLI 参数可能不同。
如果 hf download 不支持某个旧参数，就不要继续复制旧教程。
```

---

## 11. KV cache 是什么？

Transformer decode 阶段每生成一个 token，都需要历史 token 的 Key 和 Value。

为了避免每次都重新计算历史 token，系统会缓存这些 Key/Value，这就是：

```text
KV cache
```

KV cache 的大小会随着以下因素增长：

```text
上下文长度
batch size
模型层数
hidden size
KV head 数量
数据类型，例如 FP16 / BF16
```

直观理解：

```text
上下文越长，KV cache 越大。
并发请求越多，KV cache 越大。
模型越大，KV cache 越大。
```

---

## 12. KV cache offloading 要解决什么问题？

GPU 显存是有限的。

例如 RTX 4090 通常只有 24GB 显存。模型权重、activation、workspace、KV cache 都要占显存。

当上下文很长、batch 很大、并发很多时，KV cache 可能成为显存瓶颈。

KV cache offloading 研究的问题是：

```text
哪些 KV cache 留在 GPU HBM？
哪些放到 CPU DRAM？
哪些放到本地 NVMe / SSD？
什么时候搬出去？
什么时候搬回来？
怎么避免搬运阻塞 decode？
怎么判断搬运比重算更划算？
怎么让 offloading 不破坏 vLLM 的调度和 batching？
```

典型路径是：

```text
GPU HBM
  ↔ CPU DRAM
  ↔ 本地 NVMe / SSD
```

不要把实验热路径放到：

```text
GPU ↔ CPU ↔ 网络文件系统 / 分布式文件系统
```

否则测到的可能是网络存储延迟和抖动，而不是 KV cache offloading 策略本身。

---

## 13. 做 vLLM 源码开发时，先改哪里？

刚开始不要直接改 CUDA kernel。

建议先从 Python 层理解和修改：

```text
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

这一阶段一般不需要 CUDA Toolkit。

建议先回答这些问题：

```text
一个 request 进入 vLLM 后，在哪里排队？
什么时候分配 KV block？
KV block 和 token 的关系是什么？
什么时候释放 KV block？
prefix cache 怎么复用？
当显存不足时，vLLM 现在怎么处理？
是否已有 swap/offload 相关路径？
```

只有当你明确需要修改下面这些内容时，才进入更底层环境：

```text
csrc/
CUDA kernel
PagedAttention kernel
FlashAttention backend
Triton kernel
torch custom op
```

---

## 14. 第一阶段的正确目标

第一阶段不要试图一次性完成所有事情。

第一阶段只做：

```text
1. 安装 NVIDIA Driver
2. nvidia-smi 能看到所有 GPU
3. 建立 micromamba 环境
4. 安装 PyTorch
5. torch.cuda.is_available() == True
6. 安装 / editable 安装 vLLM
7. 跑通一个小模型
8. 再跑目标 Qwen 模型
```

不要一开始就做：

```text
装 CUDA Toolkit
编译 FlashAttention
完整编译 vLLM CUDA/C++ extension
跑最大模型
跑超长上下文
改底层 kernel
```

---

## 15. 一句话给新人解释

可以这样给新人讲：

```text
我们不是简单地“装一个大模型环境”，而是在搭建一个可以做 vLLM / KV cache offloading 研究的 GPU 系统环境。第一步是安装 NVIDIA Driver，因为 driver 是 Linux 使用 GPU 的基础。CUDA Toolkit、PyTorch CUDA wheel、vLLM、FlashAttention、模型权重和 KV cache offloading 是后面的不同层次，不能一开始混在一起装。
```

---

# Part II：GitHub 仓库组织、文件边界与环境迁移

> 这一部分后续继续完善。原则上：代码、脚本、配置模板、文档进 GitHub；模型权重、环境目录、大日志、大结果文件不进 GitHub。
