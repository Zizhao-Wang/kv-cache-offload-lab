# Step 3 Prompt: KV Cache Profiling Design Based on Code Reading

You are my **vLLM / KV Cache Offloading Research Assistant**.

This is **Step 3** of the workflow.
Your job is to design a profiling strategy based on Step 1 and Step 2.

Do **not** implement the design yet.

Your final report must be written in **Chinese**.

---

## Hard Rules

1. Do **not** modify vLLM source code.
2. Do **not** create patches.
3. Do **not** execute Step 4.
4. Base this design on Step 1 and Step 2 outputs.
5. The design must be logical, architecture-aware, and specific.
6. The design must include clickable links to the functions that would be modified later.
7. The report must include diagrams or tables.
8. This prompt is in English, but your output must be Chinese.

---

## Preconditions

Step 1 and Step 2 must be DONE.

Check:

```bash
cat /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/STATUS.md
test -f /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/01_architecture/step1_vllm_architecture.md
test -f /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/02_kv_cache_offload_reading/step2_kv_cache_offload_code_reading.md
```

If either Step 1 or Step 2 is missing, stop and ask me to finish them first.

---

## Output File

Create:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/03_profile_design/step3_profile_design.md
```

---

## Design Goals

Design profiling for two research questions.

### Research Question A: recompute cost

If KV cache is evicted, freed, or lost due to preemption, vLLM may need to recompute tokens later. We need to understand:

1. where recompute candidates appear,
2. how many tokens/blocks may need recomputation,
3. how prefix cache affects recomputation,
4. how long model forward/recompute takes,
5. when recomputation may be cheaper than reload.

### Research Question B: KV offload I/O characteristics

If KV blocks were offloaded to CPU memory or SSD, we need to understand:

1. number of candidate blocks,
2. block size and estimated bytes,
3. allocation/free/reuse timing,
4. prefix reuse and reuse distance,
5. store/load pattern,
6. whether the pattern is sequential, random, or batchable,
7. what metadata should go to fast tier,
8. what tensor payload should go to slow tier.

---

## Required Output Structure

The report must be in Chinese and contain:

```markdown
# Step 3: KV Cache Profiling 设计方案

## 0. 本步骤结论
Short summary of the recommended profiling design.

## 1. 设计原则
Explain:
- why we should not modify CUDA kernels now
- why we should not implement SSD offload immediately
- why we should first collect traces
- why each hook must be incremental and verifiable

## 2. 需要回答的 research questions
Explain recompute cost and offload I/O questions.

## 3. Profiling 事件总览
Define JSONL event types such as:
- engine_start
- scheduler_step_start/end
- request_arrive
- kv_allocate
- kv_free
- prefix_cache_hit/miss
- block_evict
- preempt
- recompute_candidate
- model_forward_timing

## 4. JSONL Schema 草案
For each event, define fields:
- ts_ns
- event_type
- request_id if available
- block_id/block_ids if available
- num_blocks
- estimated_bytes
- source_component
- detail

## 5. Recompute Cost Profiling 设计
Explain which functions to instrument later, why, and what to collect.
Include clickable links to candidate source files/functions.

## 6. KV Offload I/O Profiling 设计
Explain which functions to instrument later, why, and what to collect.
Include clickable links to candidate source files/functions.

## 7. Fast tier / slow tier 的数据模型
Explain:
fast tier = metadata/key/index/control information
slow tier = large KV tensor payload
Explain how profiling prepares for this design.

## 8. 推荐 patch 顺序
Give an incremental sequence:
1. trace writer only
2. scheduler step logging
3. KV allocation logging
4. KV free logging
5. prefix cache logging
6. preemption/recompute logging
7. model forward timing
8. trace analyzer

## 9. 每个 hook 的风险等级
Use a table:
- hook
- target function
- reason
- risk
- how to verify
- how to revert

## 10. Benchmark / workload 初步选择
Explain why not to randomly pick one dataset.
Classify workloads:
- low KV pressure
- medium KV pressure
- high KV pressure
- prefix reuse workloads
- agent/code/reasoning workloads
Mention synthetic prompts, ShareGPT-style traces, LongBench, Needle-in-a-haystack, RAG repeated-prefix, SWE-bench/code-agent style workloads.

## 11. 现在不应该做什么
List things to avoid:
- real SSD offload
- CUDA kernel changes
- scheduler policy changes
- tensor layout changes
- huge all-in-one patch
- logging prompts or huge tensors

## 12. Step 4 应该如何实现
Describe that Step 4 will provide manual patch-by-patch instructions, not one giant implementation.
```

---

## Important Design Style

The report should make it easy for me to understand and later implement manually.

For each proposed hook, explain:

```text
为什么这个 hook 在架构上合理？
它能回答哪个 research question？
它会记录什么？
它在哪里加？
风险是什么？
加完后怎么验证？
```

---

## Completion Criteria

Step 3 is DONE only if:

1. The output file exists.
2. The report is in Chinese.
3. It is based on Step 1 and Step 2.
4. It includes JSONL event design.
5. It includes hook design with clickable code links.
6. It includes benchmark/workload plan.
7. It does not modify vLLM source code.
8. It updates `STATUS.md`: Step 3 = DONE.
9. It tells me the next step is Step 4.
