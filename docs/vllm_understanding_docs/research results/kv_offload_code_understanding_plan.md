# vLLM KV Cache Offloading Code Understanding Plan

> vLLM fork: `ca97f7b9b` (branch `kv-offload-dev`)
> 上游: `https://github.com/vllm-project/vllm.git`
> 分析日期: 2026-05-01

---

## 1. 当前 vLLM 代码结构理解

### 1.1 请求生命周期

```
用户请求 (EngineCoreRequest)
  → EngineCore.add_request()
    → Scheduler.add_request() → 加入 waiting 队列
  → EngineCore.step()
    → Scheduler.schedule()
      → 遍历 running 队列:
        → KVCacheManager.allocate_slots() → 分配新 KV block
        → 如果 block 不足 → _preempt_request() → 释放 block, 请求回到 waiting
      → 遍历 waiting 队列:
        → KVCacheManager.get_computed_blocks() → prefix cache 查找
        → KVCacheManager.allocate_slots() → 分配 block
    → ModelExecutor.execute_model()
      → GPUModelRunner.execute_model()
        → _update_states() → 更新 batch 状态
        → _model_forward() → 实际模型前向推理
    → Scheduler.update_from_output()
      → 处理完成的请求 → _free_request() → _free_blocks()
```

### 1.2 KV Cache 管理层次

```
KVCacheManager (vllm/v1/core/kv_cache_manager.py)
  ├── KVCacheCoordinator (vllm/v1/core/kv_cache_coordinator.py)
  │     ├── UnitaryKVCacheCoordinator (单一 KV cache group)
  │     └── HybridKVCacheCoordinator (混合 KV cache group)
  │           └── SingleTypeKVCacheManager (vllm/v1/core/single_type_kv_cache_manager.py)
  │                 ├── FullAttentionManager
  │                 ├── SlidingWindowManager
  │                 ├── MambaManager
  │                 └── ...
  └── BlockPool (vllm/v1/core/block_pool.py)
        ├── FreeKVCacheBlockQueue (双向链表, LRU 驱逐)
        ├── cached_block_hash_to_block (prefix cache 哈希表)
        └── KVCacheMetricsCollector (采样式 metrics)
```

### 1.3 已有的 KV Offload 模块

vLLM 已经有两套 KV offload 实现:

1. **`vllm/v1/kv_offload/`** — 通用 offload 框架
   - `base.py`: `OffloadingSpec`, `OffloadingManager`, `OffloadingEvent` 抽象
   - `factory.py`: `OffloadingSpecFactory` 注册/创建 offload spec
   - `cpu/`: CPU offload 具体实现 (LRU/ARC 策略, 共享内存区域)
   - `reuse_manager.py`: `FilterReusedOffloadingManager` — 基于重用频率过滤 store

2. **`vllm/v1/simple_kv_offload/`** — 简化版 CPU offload
   - `manager.py`: `SimpleCPUOffloadScheduler` — scheduler 侧调度
   - `worker.py`: `SimpleCPUOffloadWorker` — worker 侧 GPU↔CPU 传输
   - `copy_backend.py`: `DmaCopyBackend` — 基于 `cuMemcpyBatchAsync` 的后台拷贝
   - `cuda_mem_ops.py`: 底层 CUDA 内存操作


---

## 2. 关键文件和函数

### 2.1 Scheduler 层

| 文件 | 类/函数 | 功能 | 与 KV offload 的关系 | 可否安全插桩 |
|------|---------|------|---------------------|-------------|
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` | 主调度循环: 遍历 running/waiting 请求, 分配 token budget | 是 preemption 和 block 分配的入口; 可在此记录每步调度决策 | ✅ 低风险 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler._preempt_request()` | 抢占请求: 释放 KV block, 重置 num_computed_tokens, 放回 waiting | **核心**: 抢占后请求需要 recompute; 可在此记录抢占事件 | ✅ 低风险 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler._free_request()` | 请求完成后释放资源 | 可记录请求完成时的 block 释放 | ✅ 低风险 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler._free_blocks()` | 调用 `kv_cache_manager.free(request)` | block 释放的直接入口 | ✅ 低风险 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler._handle_invalid_blocks()` | 处理 KV load 失败的 block, 支持 recompute 策略 | 已有 recompute 逻辑的参考 | ✅ 低风险 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.update_from_output()` | 处理模型输出, 更新请求状态 | 可在此记录每步完成的 token 数 | ✅ 低风险 |

### 2.2 KV Cache Manager 层

| 文件 | 类/函数 | 功能 | 与 KV offload 的关系 | 可否安全插桩 |
|------|---------|------|---------------------|-------------|
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` | 为请求分配 KV block slot | **核心**: block 分配入口, 可记录分配事件 | ✅ 低风险 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.get_computed_blocks()` | prefix cache 查找 | 可记录 cache hit/miss | ✅ 低风险 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.free()` | 释放请求的所有 block | block 释放入口 | ✅ 低风险 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.evict_blocks()` | 驱逐指定 block | 驱逐事件记录点 | ✅ 低风险 |

### 2.3 Block Pool 层

| 文件 | 类/函数 | 功能 | 与 KV offload 的关系 | 可否安全插桩 |
|------|---------|------|---------------------|-------------|
| `vllm/v1/core/block_pool.py` | `BlockPool.get_new_blocks()` | 从 free queue 分配 block | **核心**: 物理 block 分配, 已有 `metrics_collector.on_block_allocated()` 回调 | ✅ 低风险 |
| `vllm/v1/core/block_pool.py` | `BlockPool.free_blocks()` | 释放 block, 减少 ref_cnt, 放回 free queue | **核心**: 物理 block 释放 | ✅ 低风险 |
| `vllm/v1/core/block_pool.py` | `BlockPool._maybe_evict_cached_block()` | 从 prefix cache 驱逐 block | **核心**: 驱逐事件, 已有 `metrics_collector.on_block_evicted()` 回调和 `KVCacheEvent` | ✅ 低风险 |
| `vllm/v1/core/block_pool.py` | `BlockPool.get_cached_block()` | prefix cache 查找 | cache hit 记录点 | ✅ 低风险 |
| `vllm/v1/core/block_pool.py` | `BlockPool.cache_full_blocks()` | 将满 block 加入 prefix cache | 新 block 缓存事件, 已有 `BlockStored` event | ✅ 低风险 |
| `vllm/v1/core/block_pool.py` | `BlockPool.touch()` | 增加 block ref_cnt (cache hit 时) | 已有 `metrics_collector.on_block_accessed()` 回调 | ✅ 低风险 |

### 2.4 已有 Metrics 基础设施

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `vllm/v1/core/kv_cache_metrics.py` | `KVCacheMetricsCollector` | 采样式 block 生命周期追踪 (birth, access, eviction) |
| `vllm/v1/core/kv_cache_metrics.py` | `BlockMetricsState` | 单个 block 的 metrics 状态 |
| `vllm/v1/metrics/stats.py` | `KVCacheEvictionEvent` | 驱逐事件数据结构 (lifetime, idle_time, reuse_gaps) |
| `vllm/v1/metrics/stats.py` | `PrefixCacheStats` | prefix cache 命中率统计 |
| `vllm/v1/metrics/stats.py` | `SchedulerStats` | 调度器统计 (包含 kv_cache_usage, eviction_events) |
| `vllm/distributed/kv_events.py` | `KVCacheEvent`, `BlockStored`, `BlockRemoved` | KV cache 事件系统 (支持 ZMQ 发布) |

### 2.5 Worker / Model Runner 层

| 文件 | 类/函数 | 功能 | 与 KV offload 的关系 | 可否安全插桩 |
|------|---------|------|---------------------|-------------|
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model()` | 完整的模型执行流程 | 可在此添加 forward 计时 | ✅ 低风险 |
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._model_forward()` | 实际调用 `self.model(...)` | **最佳计时点**: 纯模型前向推理 | ✅ 低风险 |
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._allocate_kv_cache_tensors()` | 分配 GPU KV cache tensor | 可获取每层 KV cache 大小 | ✅ 低风险 |
| `vllm/v1/worker/gpu_worker.py` | `Worker.execute_model()` | worker 级别的模型执行 | 包含 KV connector 交互 | ⚠️ 中风险 |

### 2.6 已有 KV Offload 模块

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `vllm/v1/kv_offload/base.py` | `OffloadingSpec`, `OffloadingManager` | offload 抽象基类 |
| `vllm/v1/kv_offload/factory.py` | `OffloadingSpecFactory` | offload spec 工厂 |
| `vllm/v1/kv_offload/cpu/manager.py` | `CPUOffloadingManager` | CPU offload 管理器 (LRU/ARC) |
| `vllm/v1/kv_offload/cpu/spec.py` | `CPUOffloadingSpec` | CPU offload 规格 |
| `vllm/v1/kv_offload/cpu/gpu_worker.py` | `SingleDirectionOffloadingHandler` | GPU 侧 offload 处理 |
| `vllm/v1/kv_offload/cpu/shared_offload_region.py` | `SharedOffloadRegion` | 共享内存区域 (mmap) |
| `vllm/v1/kv_offload/reuse_manager.py` | `FilterReusedOffloadingManager` | 基于重用频率过滤 store |
| `vllm/v1/simple_kv_offload/manager.py` | `SimpleCPUOffloadScheduler` | 简化版 scheduler 侧调度 |
| `vllm/v1/simple_kv_offload/worker.py` | `SimpleCPUOffloadWorker` | 简化版 worker 侧传输 |
| `vllm/v1/simple_kv_offload/copy_backend.py` | `DmaCopyBackend` | cuMemcpyBatchAsync 后台拷贝 |


---

## 3. Recompute Profiling 方案

### 3.1 当前 vLLM 是否有 recomputation / preemption 逻辑?

**有。** 当前 vLLM v1 有完整的 preemption 逻辑, 但 **没有独立的 recomputation 模块**。

Preemption 的工作方式:
1. `Scheduler.schedule()` 中, 当 running 请求需要新 block 但 free block 不足时, 会调用 `_preempt_request()` 抢占低优先级请求
2. `_preempt_request()` 释放被抢占请求的所有 KV block, 将 `num_computed_tokens` 重置为 0, 请求回到 waiting 队列
3. 被抢占的请求下次被调度时, 会从头开始 prefill (即 recompute 所有 token)
4. 如果 prefix caching 开启, 被抢占请求的 block 可能仍在 prefix cache 中, 下次调度时可以命中 cache 避免部分 recompute

**关键发现**: vLLM v1 的 preemption 策略是 **recompute-only** (没有 swap-to-CPU 选项)。被抢占的请求总是需要重新计算, 不会将 KV cache swap 到 CPU。这与 v0 引擎不同 (v0 有 SWAP 和 RECOMPUTE 两种 preemption mode)。

此外, `_handle_invalid_blocks()` 和 `_update_requests_with_invalid_blocks()` 实现了 KV load 失败时的 recompute 逻辑 (通过 `kv_load_failure_policy: "recompute"` 配置)。

### 3.2 Preemption 发生的位置

```python
# vllm/v1/core/sched/scheduler.py
Scheduler.schedule()  # 主调度循环
  → 当 allocate_slots() 返回 None (block 不足):
    → _preempt_request(preempted_req, scheduled_timestamp)
      → kv_cache_manager.free(request)     # 释放所有 KV block
      → request.status = PREEMPTED         # 标记为被抢占
      → request.num_computed_tokens = 0    # 重置计算进度
      → request.num_preemptions += 1       # 抢占计数
      → waiting.prepend_request(request)   # 放回 waiting 队列头部
```

### 3.3 Recomputation 触发的位置

Recomputation 不是一个显式的函数调用, 而是隐式发生的:

1. 被抢占的请求 (`status == PREEMPTED`) 重新进入 waiting 队列
2. 下次 `schedule()` 调度到该请求时:
   - `get_computed_blocks()` 查找 prefix cache → 可能部分命中
   - `allocate_slots()` 分配新 block
   - 请求进入 running 队列, `num_computed_tokens` 从 0 (或 prefix cache hit 的位置) 开始
3. `GPUModelRunner.execute_model()` 执行模型前向推理 → 这就是 recomputation

### 3.4 计时器应该添加的位置

| 位置 | 要测量的内容 | 具体方法 |
|------|-------------|---------|
| `Scheduler._preempt_request()` | 抢占事件: request_id, num_computed_tokens (被丢弃的), num_preemptions | 记录 JSONL 事件 |
| `Scheduler.schedule()` 中调度 PREEMPTED 请求时 | recompute 开始: request_id, num_tokens_to_recompute, prefix_cache_hit_tokens | 记录 JSONL 事件 |
| `GPUModelRunner._model_forward()` | 模型前向推理耗时 | `torch.cuda.Event` 计时 |
| `GPUModelRunner.execute_model()` | 完整执行耗时 (含 preprocess/postprocess) | `time.monotonic()` 计时 |
| `KVCacheManager.get_computed_blocks()` | prefix cache 查找结果 | 已有 `PrefixCacheStats`, 可扩展 |

### 3.5 可收集的 Metrics

- 每次 preemption: `{event: "preempt", request_id, num_computed_tokens_lost, num_preemptions, timestamp}`
- 每次 recompute (被抢占请求重新调度): `{event: "recompute_start", request_id, total_tokens, prefix_cache_hit_tokens, tokens_to_recompute, timestamp}`
- 每次 model forward: `{event: "forward", num_tokens, num_reqs, elapsed_ms, is_prefill, timestamp}`

### 3.6 不确定的地方

1. **prefix cache 对 recompute 的影响**: 被抢占请求的 block 可能在 prefix cache 中存活 (如果没有被其他请求驱逐), 这会减少实际 recompute 量。需要实际测量才能知道 prefix cache 的命中率。
2. **model forward 计时的精度**: `_model_forward()` 内部可能有 CUDA kernel 异步执行, 需要用 `torch.cuda.Event` 而非 `time.monotonic()` 来精确计时。
3. **多 GPU 场景**: 在 tensor parallel 或 pipeline parallel 下, preemption 和 recompute 的行为可能不同, 需要进一步确认。

---

## 4. KV Offload I/O Profiling 方案

### 4.1 KV Block 分配位置

```
BlockPool.get_new_blocks(num_blocks)          # 物理 block 分配
  → FreeKVCacheBlockQueue.popleft_n(n)        # 从 free queue 取 block
  → _maybe_evict_cached_block(block)          # 如果 block 有 cache, 先驱逐
  → block.ref_cnt += 1                        # 增加引用计数
  → metrics_collector.on_block_allocated()    # 已有回调!
```

调用链:
```
Scheduler.schedule()
  → KVCacheManager.allocate_slots()
    → KVCacheCoordinator.allocate_new_blocks()
      → SingleTypeKVCacheManager.allocate_new_blocks()
        → BlockPool.get_new_blocks()
```

### 4.2 KV Block 释放位置

```
BlockPool.free_blocks(ordered_blocks)         # 物理 block 释放
  → block.ref_cnt -= 1                        # 减少引用计数
  → 如果 ref_cnt == 0: 放回 free_block_queue  # LRU 尾部
```

调用链:
```
Scheduler._preempt_request()  或  Scheduler._free_request()
  → KVCacheManager.free(request)
    → KVCacheCoordinator.free(request_id)
      → SingleTypeKVCacheManager.free(request_id)
        → BlockPool.free_blocks(blocks)  # 逆序释放, 尾部 block 先驱逐
```

### 4.3 KV Block 重用位置

```
BlockPool.get_cached_block(block_hash, group_ids)  # prefix cache 查找
  → cached_block_hash_to_block.get_one_block()     # 哈希表查找
  → 返回已缓存的 block (cache hit)

BlockPool.touch(blocks)                             # 增加 ref_cnt (重用)
  → 如果 ref_cnt == 0: 从 free queue 移除          # 不再是驱逐候选
  → block.ref_cnt += 1
  → metrics_collector.on_block_accessed()           # 已有回调!
```

### 4.4 Prefix Cache Hit/Miss 位置

```
KVCacheManager.get_computed_blocks(request)
  → KVCacheCoordinator.find_longest_cache_hit(block_hashes, max_length)
    → SingleTypeKVCacheManager.find_longest_cache_hit()
      → BlockPool.get_cached_block()  # 逐 block 查找
      → BlockPool.touch()             # hit 时增加 ref_cnt
  → PrefixCacheStats.record(num_tokens, num_hits, preempted)  # 已有统计!
```

### 4.5 GPU / CPU KV Cache Block 管理

**GPU 侧:**
- `BlockPool` 管理所有 GPU KV cache block
- `FreeKVCacheBlockQueue` 维护 free block 的 LRU 双向链表
- `cached_block_hash_to_block` 维护 prefix cache 哈希表
- 物理 tensor 由 `GPUModelRunner._allocate_kv_cache_tensors()` 分配

**CPU 侧 (已有的 offload 实现):**
- `kv_offload/cpu/manager.py` → `CPUOffloadingManager`: 管理 CPU 侧 block, 支持 LRU/ARC 策略
- `kv_offload/cpu/shared_offload_region.py` → `SharedOffloadRegion`: 通过 mmap 共享内存分配 CPU tensor
- `simple_kv_offload/manager.py` → `SimpleCPUOffloadScheduler`: 简化版, scheduler 侧决定 store/load
- `simple_kv_offload/worker.py` → `SimpleCPUOffloadWorker`: worker 侧执行 GPU↔CPU DMA 拷贝

### 4.6 vLLM 已有的 Swap / CPU Offload / Block Migration 逻辑

**已有:**
1. ✅ `kv_offload/` 框架: 完整的 GPU→CPU offload 抽象 (OffloadingSpec, OffloadingManager)
2. ✅ `simple_kv_offload/`: 简化版 CPU offload (DMA copy, stream-based async)
3. ✅ `kv_offload/cpu/`: CPU offload 具体实现 (LRU/ARC 策略, 共享内存)
4. ✅ `kv_offload/reuse_manager.py`: 基于重用频率的 store 过滤
5. ✅ KV cache event 系统: `BlockStored`, `BlockRemoved`, `AllBlocksCleared` (支持 ZMQ 发布)
6. ✅ KV connector 框架: 支持 P/D (prefill/decode) 分离, 远程 KV 传输

**没有:**
1. ❌ SSD/NVMe offload
2. ❌ 异构存储分层 (fast tier / slow tier)
3. ❌ 基于 block 热度的自适应 offload 策略
4. ❌ Preemption 时的 swap-to-CPU (v1 只有 recompute)

### 4.7 最佳日志记录点

| 记录点 | 函数 | 可获取的信息 |
|--------|------|-------------|
| Block 分配 | `BlockPool.get_new_blocks()` | block_id, num_blocks, 是否驱逐了 cached block |
| Block 释放 | `BlockPool.free_blocks()` | block_id, ref_cnt 变化, 是否真正释放 |
| Block 驱逐 | `BlockPool._maybe_evict_cached_block()` | block_id, block_hash, 驱逐原因 |
| Cache hit | `BlockPool.get_cached_block()` | block_hash, hit/miss |
| Cache store | `BlockPool.cache_full_blocks()` | block_hash, request_id, token_ids |
| Block touch | `BlockPool.touch()` | block_id, ref_cnt |
| 请求分配 | `KVCacheManager.allocate_slots()` | request_id, num_new_tokens, num_blocks_allocated |
| 请求释放 | `KVCacheManager.free()` | request_id, 释放的 block 数 |
| Preemption | `Scheduler._preempt_request()` | request_id, num_computed_tokens, num_preemptions |
| Prefix cache 统计 | `KVCacheManager.get_computed_blocks()` | request_id, num_tokens, num_hits |

### 4.8 可收集的信息

每个 JSONL 事件可包含:

```json
{
  "timestamp_ns": 1234567890,
  "event": "block_alloc|block_free|block_evict|cache_hit|cache_miss|preempt|...",
  "request_id": "req-xxx",
  "block_id": 42,
  "block_hash": "0xabcdef",
  "ref_cnt": 1,
  "num_blocks": 5,
  "estimated_bytes": 131072,
  "num_computed_tokens": 1024,
  "num_total_tokens": 2048,
  "kv_cache_usage": 0.85,
  "num_free_blocks": 100,
  "num_running_reqs": 8,
  "num_waiting_reqs": 3
}
```

**KV block 大小估算方法:**
- `KVCacheSpec.page_size_bytes` 属性提供每个 block 的字节数
- 对于标准 attention: `2 * block_size * num_kv_heads * head_size * dtype_size`
- 例如: block_size=16, num_kv_heads=8, head_size=128, fp16 → 2 * 16 * 8 * 128 * 2 = 65536 bytes/block



---

## 5. 建议修改点

> ⚠️ 本节仅列出未来的修改计划, 当前不修改任何代码。

### 5.1 JSONL Trace Writer (新增文件)

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/core/kv_trace_writer.py` (新建) |
| 添加内容 | `KVTraceWriter` 类: 异步写入 JSONL 文件, 支持 buffer flush, 环境变量控制开关 |
| 原因 | 统一的 trace 输出, 避免在多个文件中重复日志逻辑 |
| 风险 | 🟢 低 — 纯新增文件, 不修改现有逻辑 |

### 5.2 BlockPool 插桩

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/core/block_pool.py` |
| 目标函数 | `get_new_blocks()`, `free_blocks()`, `_maybe_evict_cached_block()`, `touch()`, `get_cached_block()` |
| 添加内容 | 在已有的 `metrics_collector` 回调旁边, 添加 `trace_writer` 调用 |
| 原因 | 记录 block 级别的分配/释放/驱逐/重用事件 |
| 风险 | 🟢 低 — 仅添加可选的日志调用, 不改变控制流 |

### 5.3 KVCacheManager 插桩

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/core/kv_cache_manager.py` |
| 目标函数 | `allocate_slots()`, `free()`, `get_computed_blocks()` |
| 添加内容 | 记录请求级别的 block 分配/释放/prefix cache hit 事件 |
| 原因 | 关联 request_id 和 block 操作 |
| 风险 | 🟢 低 — 仅添加日志 |

### 5.4 Scheduler 插桩

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/core/sched/scheduler.py` |
| 目标函数 | `_preempt_request()`, `schedule()` (调度 PREEMPTED 请求时) |
| 添加内容 | 记录 preemption 事件和 recompute 开始事件 |
| 原因 | 追踪 preemption 导致的 recompute 成本 |
| 风险 | 🟢 低 — 仅添加日志 |

### 5.5 Model Forward 计时

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/worker/gpu_model_runner.py` |
| 目标函数 | `_model_forward()` 或 `execute_model()` |
| 添加内容 | `torch.cuda.Event` 计时, 记录每步 forward 耗时 |
| 原因 | 测量 recompute 的实际 GPU 时间成本 |
| 风险 | 🟡 中 — CUDA event 可能有微小性能影响; 需要确保不影响 CUDA graph |

### 5.6 KV Cache 大小信息收集

| 项目 | 内容 |
|------|------|
| 目标文件 | `vllm/v1/worker/gpu_model_runner.py` |
| 目标函数 | `initialize_kv_cache()` 或 `_allocate_kv_cache_tensors()` |
| 添加内容 | 在初始化时记录每层 KV cache 的 page_size_bytes, num_blocks, total_bytes |
| 原因 | 为后续 offload 估算提供基础数据 |
| 风险 | 🟢 低 — 仅在初始化时记录一次 |



---

## 6. 第一阶段最小实现建议

### Phase 1: Trace Infrastructure (预计 1-2 天)

1. **新建 `KVTraceWriter`** (`vllm/v1/core/kv_trace_writer.py`)
   - 环境变量 `VLLM_KV_TRACE_PATH` 控制输出路径 (不设置则不记录)
   - 异步 buffered JSONL 写入
   - 支持 `flush()` 和 `close()`
   - 每条记录包含: timestamp_ns, event_type, 以及事件特定字段

2. **BlockPool 插桩** (`vllm/v1/core/block_pool.py`)
   - 在 `__init__()` 中接受可选的 `trace_writer` 参数
   - 在 `get_new_blocks()` 中记录 `block_alloc` 事件
   - 在 `free_blocks()` 中记录 `block_free` 事件
   - 在 `_maybe_evict_cached_block()` 中记录 `block_evict` 事件
   - 在 `get_cached_block()` 中记录 `cache_hit` / `cache_miss` 事件
   - 在 `touch()` 中记录 `block_touch` 事件

3. **Scheduler 插桩** (`vllm/v1/core/sched/scheduler.py`)
   - 在 `_preempt_request()` 中记录 `preempt` 事件
   - 在 `schedule()` 中调度 PREEMPTED 请求时记录 `recompute_start` 事件
   - 在 `_free_request()` 中记录 `request_complete` 事件

4. **Model Forward 计时** (`vllm/v1/worker/gpu_model_runner.py`)
   - 在 `_model_forward()` 前后添加 `torch.cuda.Event` 计时
   - 记录 `forward_timing` 事件 (num_tokens, elapsed_ms)
   - 通过环境变量 `VLLM_KV_TRACE_FORWARD_TIMING=1` 控制开关

5. **Smoke Test 脚本** (`repos/kv-cache-offload-lab/scripts/smoke_test_trace.py`)
   - 启动 vLLM 服务, 发送几个请求
   - 验证 JSONL trace 文件生成
   - 验证事件格式正确
   - 打印基本统计 (block 分配/释放次数, preemption 次数等)

### Phase 1 的预期输出

```jsonl
{"ts_ns":1234567890,"event":"kv_cache_init","num_blocks":1024,"block_size":16,"page_size_bytes":65536}
{"ts_ns":1234567891,"event":"block_alloc","block_ids":[10,11,12],"request_id":"req-001"}
{"ts_ns":1234567892,"event":"cache_hit","block_hash":"0xabc","request_id":"req-002","num_hit_blocks":5}
{"ts_ns":1234567893,"event":"preempt","request_id":"req-001","tokens_lost":512,"preemption_count":1}
{"ts_ns":1234567894,"event":"recompute_start","request_id":"req-001","total_tokens":512,"cache_hit_tokens":256}
{"ts_ns":1234567895,"event":"forward","num_tokens":128,"num_reqs":4,"elapsed_ms":12.5}
{"ts_ns":1234567896,"event":"block_free","block_ids":[10,11,12],"request_id":"req-001"}
{"ts_ns":1234567897,"event":"block_evict","block_id":10,"block_hash":"0xabc"}
```



---

## 7. 不建议现在做的事情

| 不建议做的事情 | 原因 |
|---------------|------|
| 修改 CUDA kernel (`vllm/kernels/`, `vllm/vllm_flash_attn/`) | 风险极高, 可能导致数值错误或性能退化; 需要深入理解 attention backend |
| 修改 scheduler 调度策略 | 会影响所有请求的调度行为, 可能导致性能退化或死锁 |
| 修改 KV tensor 的内存布局 | 会影响所有 attention backend, 需要同步修改 CUDA kernel |
| 实现真正的 SSD offload | 需要先完成 profiling, 理解 I/O 模式后再设计; 过早实现可能方向错误 |
| 记录 prompt 内容或大 tensor | 会导致 trace 文件巨大, 可能泄露隐私数据 |
| 修改 `kv_offload/` 或 `simple_kv_offload/` 的现有逻辑 | 这些是上游维护的功能, 修改后难以合并上游更新 |
| 在 CUDA graph 路径中添加 Python 回调 | 会破坏 CUDA graph capture, 导致性能严重退化 |
| 修改 `forward_context.py` 中的 `set_forward_context()` | 这是所有 attention backend 的核心上下文, 修改风险高 |
| 在热路径 (每个 token 执行一次的代码) 中添加重量级日志 | 会显著影响推理性能; 应使用采样或 batch 级别的日志 |

---

## 8. 需要我确认的问题

### 8.1 关于 Profiling 范围

1. **是否需要同时支持 `kv_offload/` 和 `simple_kv_offload/` 两套 offload 框架的 profiling?**
   - 两套框架的架构不同, 插桩点也不同
   - 建议: 先只关注 `simple_kv_offload/` (更简单, 更容易理解)

2. **Trace 的粒度**: 是否需要记录每个 block 的每次操作, 还是只记录 batch 级别的统计?
   - 每个 block 的记录量可能很大 (每步可能有数百个 block 操作)
   - 建议: 默认 batch 级别, 可选 block 级别 (通过环境变量控制)

3. **是否需要记录 model forward 中每一层的耗时?**
   - 需要修改 `model_executor/` 中的模型代码, 风险较高
   - 建议: 先只记录整体 forward 耗时, 后续按需添加

### 8.2 关于已有 Offload 框架

4. **是否计划在已有的 `kv_offload/` 框架上扩展, 还是独立实现?**
   - 已有框架已经有 CPU offload 的完整抽象
   - 如果要做 SSD offload, 可以注册新的 `OffloadingSpec` (如 `SSDOffloadingSpec`)
   - 建议: 复用已有框架, 避免重复造轮子

5. **`simple_kv_offload/` vs `kv_offload/` — 哪个更适合作为研究基础?**
   - `simple_kv_offload/` 更简单, 但功能有限
   - `kv_offload/` 更完整, 有 LRU/ARC 策略和重用过滤
   - 需要你确认研究方向

### 8.3 关于异构存储

6. **fast tier / slow tier 的分界线在哪里?**
   - 是按 block 的访问频率分层, 还是按 block 的类型 (key vs value) 分层?
   - 这会影响 profiling 需要收集的信息

7. **是否需要考虑 ZNS SSD 的特殊 I/O 模式?**
   - ZNS 需要顺序写入, 这会影响 offload 策略的设计
   - 如果需要, profiling 阶段就应该记录写入模式 (顺序 vs 随机)

### 8.4 关于测试环境

8. **当前机器的 GPU 型号和显存大小?**
   - 这会影响 KV cache 的 block 数量和 offload 的必要性
   - 从目录名 `local_4090.example.env` 推测可能是 RTX 4090 (24GB)

9. **是否有可用的 NVMe SSD 用于后续的 offload 实验?**
   - 如果有, 需要了解其型号、容量、顺序/随机读写性能

10. **首选的测试模型是什么?**
    - 目录中有 `Qwen3.6-27B` 和 `gemma-4-E4B-it`
    - 27B 模型在 24GB GPU 上会有较大的 KV cache 压力, 适合测试 offload
