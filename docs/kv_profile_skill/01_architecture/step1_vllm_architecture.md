# Step 1: vLLM 整体架构解读

> 基于 vLLM fork `ca97f7b9b`，分支 `kv-offload-dev`
> vLLM 版本: `0.1.dev16209+gca97f7b9b`
> 环境: Python 3.12.13, torch 2.11.0+cu130, 2×RTX 4090

---

## 0. 本步骤结论

vLLM 不是一个有单一 `main()` 入口的程序。它是一个**分层架构**的推理引擎，核心思路是：

1. **用户请求** 通过 Python API 或 OpenAI 兼容服务器进入系统
2. **Engine** 负责请求的接收和生命周期管理
3. **Scheduler** 决定每一步哪些请求参与计算、分配多少 token 预算
4. **KV Cache Manager** 管理 GPU 上的 KV 缓存块（block）的分配、释放和复用
5. **Worker / GPUModelRunner** 执行实际的模型前向计算
6. **输出处理** 将模型输出转换为用户可读的结果

理解 vLLM 的关键是理解这个**请求驱动的循环**：`schedule() → execute_model() → update_from_output()`。

---

## 1. vLLM 有没有一个 main 函数？

**没有。** vLLM 有多种使用方式，每种方式有不同的入口：

| 使用方式 | 入口文件 | 入口类/函数 |
|---------|---------|------------|
| 离线 Python API | [`vllm/entrypoints/llm.py`](../../../../vllm/vllm/entrypoints/llm.py) | `LLM` 类，第 106 行 |
| OpenAI 兼容服务器 | [`vllm/entrypoints/openai/api_server.py`](../../../../vllm/vllm/entrypoints/openai/api_server.py) | `build_async_engine_client()`，第 80 行 |
| CLI 命令 `vllm serve` | [`vllm/entrypoints/cli/serve.py`](../../../../vllm/vllm/entrypoints/cli/serve.py) | `ServeSubcommand` 类 |

但无论哪种入口，最终都会走到同一个核心引擎：

- 离线模式 → `LLMEngine` → `EngineCoreClient` → `EngineCore`
- 服务器模式 → `AsyncLLM` → `EngineCoreClient` → `EngineCore`

---

## 2. vLLM 的核心心智模型

```
用户请求 (prompt + sampling params)
  │
  ▼
┌─────────────────────────────────┐
│  入口层 (Entrypoints)           │  LLM.generate() / OpenAI API
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  引擎层 (Engine / EngineCore)   │  请求接收、生命周期管理
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  调度层 (Scheduler)             │  决定哪些请求参与本步计算
│  ├─ running 队列                │  正在运行的请求
│  ├─ waiting 队列                │  等待调度的请求
│  └─ KVCacheManager              │  KV 缓存块分配/释放
└──────────────┬──────────────────┘
               │  SchedulerOutput
               ▼
┌─────────────────────────────────┐
│  执行层 (Executor → Worker)     │  分布式执行管理
│  └─ GPUModelRunner              │  实际模型前向计算
│     ├─ prepare_inputs()         │  准备输入张量
│     ├─ model(**inputs)          │  模型前向传播
│     └─ sample()                 │  采样输出 token
└──────────────┬──────────────────┘
               │  ModelRunnerOutput
               ▼
┌─────────────────────────────────┐
│  输出处理 (Output Processing)   │  detokenize、流式输出
└─────────────────────────────────┘
```

**核心循环**（在 `EngineCore.step()` 中）：

```
while 有未完成的请求:
    scheduler_output = scheduler.schedule()        # 调度
    model_output = executor.execute_model(...)     # 执行
    outputs = scheduler.update_from_output(...)    # 更新状态
```

---

## 3. Offline Python API 请求路径

当用户调用 `LLM.generate()` 时：

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-1.7B")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=100))
```

调用链：

1. [`LLM.generate()`](../../../../vllm/vllm/entrypoints/llm.py) 第 446 行
   - 验证参数，调用 `_run_completion()`
2. [`LLM._run_completion()`](../../../../vllm/vllm/entrypoints/llm.py) 第 1625 行
   - 调用 `_add_completion_requests()` 将请求加入引擎
   - 调用 `_run_engine()` 驱动引擎循环
3. [`LLM._run_engine()`](../../../../vllm/vllm/entrypoints/llm.py) 第 1836 行
   - `while llm_engine.has_unfinished_requests():`
   - `step_outputs = llm_engine.step()`
4. [`LLMEngine.step()`](../../../../vllm/vllm/v1/engine/llm_engine.py) 第 47 行
   - 通过 `EngineCoreClient` 调用 `EngineCore.step()`

> 绝对路径: `/home/jeff-wang/vllm_lab/repos/vllm/vllm/entrypoints/llm.py`

---

## 4. OpenAI Server 请求路径

当用户通过 HTTP 发送请求时：

```bash
curl http://localhost:8000/v1/completions -d '{"model":"...","prompt":"Hello"}'
```

调用链：

1. [`api_server.py`](../../../../vllm/vllm/entrypoints/openai/api_server.py) 第 72 行
   - FastAPI 应用接收 HTTP 请求
2. `build_async_engine_client()` 第 80 行
   - 创建 `AsyncLLM` 实例作为引擎客户端
3. [`AsyncLLM`](../../../../vllm/vllm/v1/engine/async_llm.py) 第 70 行
   - 异步引擎包装器，内部使用 `EngineCoreClient`
4. `EngineCoreClient` → `EngineCore`
   - 最终走到同一个 `EngineCore.step()` 循环

> 绝对路径: `/home/jeff-wang/vllm_lab/repos/vllm/vllm/entrypoints/openai/api_server.py`

---

## 5. Engine / EngineCore 做什么

[`EngineCore`](../../../../vllm/vllm/v1/engine/core.py) 是 vLLM 的核心引擎（第 91 行）。

### 初始化阶段 (`__init__`)

1. 创建 `ModelExecutor`（管理 GPU worker）
2. 调用 `_initialize_kv_caches()` — 分析 GPU 内存，决定 KV 缓存大小
3. 创建 `Scheduler`（调度器）
4. 设置批处理队列（pipeline parallelism 用）

### 运行阶段 (`step()`)

[`EngineCore.step()`](../../../../vllm/vllm/v1/engine/core.py) 第 402 行：

```python
def step(self):
    scheduler_output = self.scheduler.schedule()           # 1. 调度
    future = self.model_executor.execute_model(...)        # 2. 执行模型
    model_output = future.result()                         # 3. 等待结果
    engine_core_outputs = self.scheduler.update_from_output(...)  # 4. 更新
    return engine_core_outputs
```

### 请求接收 (`add_request()`)

[`EngineCore.add_request()`](../../../../vllm/vllm/v1/engine/core.py) 第 315 行：
- 验证 request_id 类型
- 调用 `self.scheduler.add_request(request)` 将请求加入调度器

> 绝对路径: `/home/jeff-wang/vllm_lab/repos/vllm/vllm/v1/engine/core.py`

---

## 6. Scheduler 做什么

[`Scheduler`](../../../../vllm/vllm/v1/core/sched/scheduler.py) 第 67 行。

### 核心职责

调度器决定**每一步（step）哪些请求参与计算**。它管理：

- **running 队列**: 正在运行的请求
- **waiting 队列**: 等待调度的新请求
- **token 预算**: `max_num_scheduled_tokens`，每步最多调度多少 token
- **KV 缓存分配**: 通过 `KVCacheManager` 为请求分配 KV block

### `schedule()` 方法的逻辑

[`Scheduler.schedule()`](../../../../vllm/vllm/v1/core/sched/scheduler.py) 第 352 行：

```
1. 先调度 RUNNING 请求（已在运行的请求优先）
   - 计算每个请求需要的新 token 数
   - 调用 kv_cache_manager.allocate_slots() 分配 KV block
   - 如果分配失败 → 抢占（preempt）低优先级请求
2. 再调度 WAITING 请求（新请求）
   - 检查 prefix cache 命中
   - 分配 KV block
   - 检查 token 预算
3. 构建 SchedulerOutput
```

### 关键概念

- **没有显式的 prefill/decode 阶段**: vLLM V1 调度器统一处理，每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`
- **Preemption（抢占）**: 当 KV 缓存不足时，释放低优先级请求的 KV block，该请求回到 waiting 队列，下次需要**重新计算**所有 KV
- **KV 缓存管理的细节留给 Step 2**

> 绝对路径: `/home/jeff-wang/vllm_lab/repos/vllm/vllm/v1/core/sched/scheduler.py`

---

## 7. Worker / GPUModelRunner 做什么

### Worker

[`Worker`](../../../../vllm/vllm/v1/worker/gpu_worker.py) 第 105 行：
- 管理单个 GPU 上的模型执行
- 持有 `GPUModelRunner` 实例

### GPUModelRunner

[`GPUModelRunner`](../../../../vllm/vllm/v1/worker/gpu/model_runner.py) 第 106 行：

核心方法 [`execute_model()`](../../../../vllm/vllm/v1/worker/gpu/model_runner.py) 第 958 行：

```
1. finish_requests()     — 清理已完成的请求
2. free_states()         — 释放 worker 侧状态
3. add_requests()        — 添加新请求到 batch
4. update_requests()     — 更新已有请求的 block table
5. prepare_inputs()      — 准备输入张量（token IDs, positions）
6. prepare_attn()        — 准备注意力元数据（block tables, slot mappings）
7. self.model(**inputs)  — 实际模型前向传播
8. 返回 hidden states
```

之后 `sample_tokens()` 方法（第 1154 行）执行采样，生成输出 token。

### Executor

[`Executor`](../../../../vllm/vllm/v1/executor/abstract.py) 第 37 行：
- 抽象基类，管理一个或多个 Worker
- 常用实现：`UniProcExecutor`（单进程）、`MultiprocExecutor`（多进程）

> 绝对路径: `/home/jeff-wang/vllm_lab/repos/vllm/vllm/v1/worker/gpu/model_runner.py`

---

## 8. 从用户请求到输出的完整流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                    用户请求进入                                    │
│  LLM.generate("Hello")  或  POST /v1/completions                │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  InputProcessor: tokenize + 多模态处理                            │
│  prompt → token_ids, mm_features                                 │
│  构建 EngineCoreRequest                                          │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  EngineCore.add_request()                                        │
│  → Scheduler.add_request()                                       │
│  请求进入 waiting 队列                                            │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  EngineCore.step() — 核心循环                                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 1. Scheduler.schedule()                                    │  │
│  │    - 遍历 running 请求，分配 KV block                       │  │
│  │    - 遍历 waiting 请求，检查 prefix cache                   │  │
│  │    - 如果 KV 不足 → preempt 低优先级请求                    │  │
│  │    - 输出: SchedulerOutput (哪些请求、多少 token)            │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 2. Executor.execute_model(scheduler_output)                │  │
│  │    → Worker.execute_model()                                │  │
│  │    → GPUModelRunner.execute_model()                        │  │
│  │      - prepare_inputs(): 构建 input_ids, positions         │  │
│  │      - prepare_attn(): 构建 block_tables, slot_mappings    │  │
│  │      - model(**inputs): 模型前向传播                        │  │
│  │    → GPUModelRunner.sample_tokens()                        │  │
│  │      - 采样输出 token                                       │  │
│  │    输出: ModelRunnerOutput                                  │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 3. Scheduler.update_from_output(model_output)              │  │
│  │    - 更新 num_computed_tokens                               │  │
│  │    - 检查停止条件 (max_tokens, EOS)                         │  │
│  │    - 标记已完成的请求                                        │  │
│  │    输出: EngineCoreOutputs                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  OutputProcessor: detokenize + 构建 RequestOutput                │
│  token_ids → text                                                │
│  返回给用户                                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. 新手应该按什么顺序读代码

| 顺序 | 文件 | 要理解什么 | 可以先跳过什么 |
|------|------|-----------|--------------|
| 1 | [`vllm/entrypoints/llm.py`](../../../../vllm/vllm/entrypoints/llm.py) | `LLM` 类如何创建引擎、`generate()` 如何调用 `_run_engine()` | 多模态、LoRA、chat 相关代码 |
| 2 | [`vllm/v1/engine/core.py`](../../../../vllm/vllm/v1/engine/core.py) | `EngineCore.__init__()` 和 `step()` 方法 | `EngineCoreProc`（ZMQ 通信）、`DPEngineCoreProc`（数据并行） |
| 3 | [`vllm/v1/core/sched/scheduler.py`](../../../../vllm/vllm/v1/core/sched/scheduler.py) | `schedule()` 方法的 running/waiting 循环 | KV connector、encoder-decoder、speculative decoding 细节 |
| 4 | [`vllm/v1/core/kv_cache_manager.py`](../../../../vllm/vllm/v1/core/kv_cache_manager.py) | `allocate_slots()` 和 `free()` 的接口 | 内部 coordinator 实现细节 |
| 5 | [`vllm/v1/worker/gpu/model_runner.py`](../../../../vllm/vllm/v1/worker/gpu/model_runner.py) | `execute_model()` 的整体流程 | CUDA graph、speculative decoding、LoRA |
| 6 | [`vllm/v1/core/block_pool.py`](../../../../vllm/vllm/v1/core/block_pool.py) | block 分配、释放、eviction 机制 | KV cache events 细节 |

---

## 10. 关键文件和函数表

| 源文件链接 | 类/函数 | 行号 | 角色 | 为什么重要 | 暂时可忽略 |
|-----------|--------|------|------|----------|-----------|
| [`llm.py`](../../../../vllm/vllm/entrypoints/llm.py) | `LLM` | 106 | 离线 API 入口 | 用户最常用的接口 | 内部 tokenizer 细节 |
| [`llm.py`](../../../../vllm/vllm/entrypoints/llm.py) | `LLM.generate()` | 446 | 生成入口 | 请求的起点 | 参数验证逻辑 |
| [`llm.py`](../../../../vllm/vllm/entrypoints/llm.py) | `LLM._run_engine()` | 1836 | 引擎驱动循环 | 理解 step 循环 | tqdm 进度条 |
| [`api_server.py`](../../../../vllm/vllm/entrypoints/openai/api_server.py) | `build_async_engine_client()` | 80 | 服务器引擎创建 | 服务器模式入口 | 中间件配置 |
| [`async_llm.py`](../../../../vllm/vllm/v1/engine/async_llm.py) | `AsyncLLM` | 70 | 异步引擎包装 | 服务器模式的引擎 | 异步 I/O 细节 |
| [`llm_engine.py`](../../../../vllm/vllm/v1/engine/llm_engine.py) | `LLMEngine` | 47 | 同步引擎包装 | 离线模式的引擎 | 统计日志 |
| [`core.py`](../../../../vllm/vllm/v1/engine/core.py) | `EngineCore` | 91 | 核心引擎 | **最重要的类** | ZMQ/Ray 通信 |
| [`core.py`](../../../../vllm/vllm/v1/engine/core.py) | `EngineCore.step()` | 402 | 核心循环 | **最重要的方法** | batch queue 逻辑 |
| [`core.py`](../../../../vllm/vllm/v1/engine/core.py) | `EngineCore.add_request()` | 315 | 请求接收 | 请求如何进入系统 | pooling 验证 |
| [`scheduler.py`](../../../../vllm/vllm/v1/core/sched/scheduler.py) | `Scheduler` | 67 | 调度器 | 决定执行什么 | Mamba/encoder 特殊逻辑 |
| [`scheduler.py`](../../../../vllm/vllm/v1/core/sched/scheduler.py) | `Scheduler.schedule()` | 352 | 调度逻辑 | **核心调度算法** | speculative decode |
| [`scheduler.py`](../../../../vllm/vllm/v1/core/sched/scheduler.py) | `Scheduler._preempt_request()` | ~980 | 抢占逻辑 | 理解 recompute | — |
| [`kv_cache_manager.py`](../../../../vllm/vllm/v1/core/kv_cache_manager.py) | `KVCacheManager` | 107 | KV 缓存管理 | block 分配接口 | coordinator 内部 |
| [`kv_cache_manager.py`](../../../../vllm/vllm/v1/core/kv_cache_manager.py) | `allocate_slots()` | 265 | 分配 KV block | 理解 KV 分配 | sliding window |
| [`block_pool.py`](../../../../vllm/vllm/v1/core/block_pool.py) | `BlockPool` | 130 | Block 池 | 物理 block 管理 | event queue |
| [`model_runner.py`](../../../../vllm/vllm/v1/worker/gpu/model_runner.py) | `GPUModelRunner` | 106 | 模型执行器 | 模型前向计算 | CUDA graph |
| [`model_runner.py`](../../../../vllm/vllm/v1/worker/gpu/model_runner.py) | `execute_model()` | 958 | 模型前向 | **实际计算发生处** | PP/DP 逻辑 |
| [`gpu_worker.py`](../../../../vllm/vllm/v1/worker/gpu_worker.py) | `Worker` | 105 | GPU Worker | 管理单 GPU | 分布式初始化 |
| [`abstract.py`](../../../../vllm/vllm/v1/executor/abstract.py) | `Executor` | 37 | 执行器基类 | Worker 管理 | Ray/多进程细节 |

---

## 11. 我如何自己复查这些结论

以下 `grep` 命令可以验证本文档中的每个关键结论：

### 验证入口类

```bash
# 找到 LLM 类定义
grep -rn "class LLM:" vllm/entrypoints/llm.py
# 预期: 第 106 行

# 找到 generate 方法
grep -rn "def generate" vllm/entrypoints/llm.py
# 预期: 第 446 行
```

### 验证引擎核心

```bash
# 找到 EngineCore 类
grep -rn "class EngineCore:" vllm/v1/engine/core.py
# 预期: 第 91 行

# 找到 step 方法
grep -rn "def step" vllm/v1/engine/core.py
# 预期: 第 402 行

# 找到 add_request 方法
grep -rn "def add_request" vllm/v1/engine/core.py
# 预期: 第 315 行
```

### 验证调度器

```bash
# 找到 Scheduler 类
grep -rn "class Scheduler" vllm/v1/core/sched/scheduler.py
# 预期: 第 67 行

# 找到 schedule 方法
grep -rn "def schedule" vllm/v1/core/sched/scheduler.py
# 预期: 第 352 行

# 找到 preempt 方法
grep -rn "_preempt_request" vllm/v1/core/sched/scheduler.py
# 预期: 约第 980 行
```

### 验证模型执行

```bash
# 找到 GPUModelRunner
grep -rn "class GPUModelRunner" vllm/v1/worker/gpu/model_runner.py
# 预期: 第 106 行

# 找到 execute_model
grep -rn "def execute_model" vllm/v1/worker/gpu/model_runner.py
# 预期: 第 958 行
```

### 验证 KV 缓存管理

```bash
# 找到 KVCacheManager
grep -rn "class KVCacheManager" vllm/v1/core/kv_cache_manager.py
# 预期: 第 107 行

# 找到 BlockPool
grep -rn "class BlockPool" vllm/v1/core/block_pool.py
# 预期: 第 130 行
```

所有命令应在 `/home/jeff-wang/vllm_lab/repos/vllm` 目录下执行。

---

## 12. 本步骤没有覆盖什么

以下内容将在后续步骤中详细分析：

| 主题 | 对应步骤 |
|------|---------|
| KV 缓存 block 的分配/释放/eviction 细节 | Step 2 |
| Prefix caching 的 hash 机制和命中/未命中路径 | Step 2 |
| Preemption 后的 recompute 机制 | Step 2 |
| 已有的 `vllm/v1/kv_offload/` offloading 框架 | Step 2 |
| KV 缓存 block 大小计算 (`page_size_bytes`) | Step 2 |
| Profiling 设计方案 | Step 3 |
| 逐 patch 实现方案 | Step 4 |

本步骤的目标是建立**整体架构的心智模型**，让你能够在后续步骤中快速定位到具体的代码位置。
