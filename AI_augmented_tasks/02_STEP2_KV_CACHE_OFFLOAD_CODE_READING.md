# Step 2 Prompt: KV Cache / Offload / Recompute Code Reading

You are my **vLLM / KV Cache Offloading Research Assistant**.

This is **Step 2** of the workflow.
Your job is to explain the vLLM KV cache, prefix cache, preemption, recompute, and existing offload-related code paths.

Your final report must be written in **Chinese**.

---

## Hard Rules

1. Do **not** modify vLLM source code.
2. Do **not** implement profiling.
3. Do **not** create patches.
4. Do **not** execute Step 3/4.
5. Base this report on Step 1's architecture understanding.
6. The report must contain clickable Markdown links to source files.
7. The report must include diagrams or ASCII flow charts.
8. Be detailed and logical, but not unnecessarily long.
9. This prompt is in English, but your output must be Chinese.

---

## Preconditions

Step 1 must be DONE.

Check:

```bash
cat /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/STATUS.md
test -f /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/01_architecture/step1_vllm_architecture.md
```

If Step 1 is not done, stop and ask me to run Step 1 first.

---

## Workspace

vLLM source repo:

```bash
/home/jeff-wang/vllm_lab/repos/vllm
```

Output file:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/02_kv_cache_offload_reading/step2_kv_cache_offload_code_reading.md
```

Relative links from this output file to vLLM source usually look like:

```markdown
[`KVCacheManager.allocate_slots()`](../../../../vllm/vllm/.../kv_cache_manager.py)
```

---

## Required Code Searches

Run:

```bash
cd /home/jeff-wang/vllm_lab/repos/vllm

rg -n "class KVCacheManager|def allocate_slots|def get_computed_blocks|def free" vllm
rg -n "class BlockPool|def get_new_blocks|def free_blocks|def touch|def get_cached_block|def cache_full_blocks" vllm
rg -n "prefix cache|PrefixCache|computed_blocks|get_computed_blocks" vllm
rg -n "preempt|recompute|swap|offload|evict" vllm
rg -n "kv_offload|simple_kv_offload|OffloadingManager|CPUOffload|CPUOffloading" vllm
rg -n "page_size_bytes|block_size|num_gpu_blocks|num_cpu_blocks" vllm
```

You may run extra searches if needed.

---

## Required Output Structure

Create:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/02_kv_cache_offload_reading/step2_kv_cache_offload_code_reading.md
```

The report must be in Chinese and contain:

```markdown
# Step 2: KV Cache / Offload / Recompute 代码解读

## 0. 本步骤结论
Short summary of how KV cache management works in this vLLM version.

## 1. Step 1 架构中，KV cache 位于哪里？
Connect back to Step 1's architecture diagram.

## 2. KV cache 生命周期总览
Use a diagram:

request scheduled
  ↓
check prefix cache
  ↓
allocate KV blocks
  ↓
model forward writes K/V
  ↓
decode reuses KV
  ↓
request finish or preempt
  ↓
free / cache / evict blocks

## 3. KVCacheManager 做什么
Explain inputs, outputs, state changes, and why it matters.
Include clickable links.

## 4. BlockPool 做什么
Explain physical/logical block allocation, free queue, ref count, reuse.
Include clickable links.

## 5. Prefix Cache 在哪里发生
Explain hit/miss path and how to verify it.
Include clickable links.

## 6. Preemption 和 Recomputation 在哪里发生
Explain whether recomputation is explicit or implicit.
Explain what happens when KV is lost or request is preempted.
Include clickable links.

## 7. vLLM 已有 offload 代码是什么
Explain `kv_offload/` and `simple_kv_offload/` if they exist in this codebase.
Explain CPU offload, swap-like behavior, and whether SSD/NVMe/ZNS offload exists.
Include clickable links.

## 8. KV block size / page size / estimated bytes 怎么看
Explain which config/spec variables matter.
Provide formula if possible.

## 9. 关键文件和函数表
Table columns:
- clickable source link
- class/function
- approximate line number
- role
- what state it changes
- why it matters for profiling
- beginner notes

## 10. 我如何自己复查这些结论
List exact `rg` commands and what each proves.

## 11. 哪些结论确定，哪些还需要后续验证
Separate confirmed facts from uncertain points.

## 12. 本步骤没有覆盖什么
Clearly say that actual profiling design is Step 3.
```

---

## Beginner Explanation Requirement

For every important function, explain:

```text
这个函数输入是什么？
输出是什么？
修改了什么状态？
和 KV cache/offload/recompute 有什么关系？
我自己要验证它，应该看什么变量？
现在看不懂的话，哪些细节可以先跳过？
```

---

## Completion Criteria

Step 2 is DONE only if:

1. The output file exists.
2. The report is in Chinese.
3. It includes diagrams.
4. It includes clickable local Markdown links to source files.
5. It explains KV allocation/free/reuse/prefix/preemption/offload paths.
6. It does not modify vLLM source code.
7. It updates `STATUS.md`: Step 2 = DONE.
8. It tells me the next step is Step 3.
