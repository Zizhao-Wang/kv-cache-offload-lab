# Step 0 Prompt: Create the Dedicated KV Profiling Skill Workspace

You are my **vLLM / KV Cache Offloading Research Assistant**.

This is **Step 0** of a multi-step skill workflow.
In this step, your only job is to create and initialize a dedicated folder for all future Markdown outputs produced by this workflow.

Your final report must be written in **Chinese**.

---

## Hard Rules

1. Do **not** modify vLLM source code.
2. Do **not** implement profiling.
3. Do **not** run benchmarks.
4. Do **not** execute Step 1/2/3/4.
5. Only create the output workspace and a small status file.
6. All final reports and generated documents must be in Chinese.
7. This prompt itself is in English, but your output must be Chinese.

---

## Workspace

Main workspace:

```bash
/home/jeff-wang/vllm_lab
```

vLLM source repo:

```bash
/home/jeff-wang/vllm_lab/repos/vllm
```

Lab repo:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab
```

Dedicated output folder for this skill:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill
```

Please create this folder and the following subfolders:

```text
docs/kv_profile_skill/
├── 00_workspace/
├── 01_architecture/
├── 02_kv_cache_offload_reading/
├── 03_profile_design/
└── 04_incremental_implementation/
```

Also create:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/STATUS.md
```

---

## STATUS.md Content

`STATUS.md` should be in Chinese and contain a simple table:

```markdown
# vLLM KV Cache Profiling Skill 状态表

| Step | Status | Output Folder | Main Output File | Notes |
|---|---|---|---|---|
| Step 0 | DONE | docs/kv_profile_skill/00_workspace | STATUS.md | 创建专用输出目录 |
| Step 1 | TODO | docs/kv_profile_skill/01_architecture | step1_vllm_architecture.md | vLLM 整体架构解读 |
| Step 2 | TODO | docs/kv_profile_skill/02_kv_cache_offload_reading | step2_kv_cache_offload_code_reading.md | KV cache / offload / recompute 代码解读 |
| Step 3 | TODO | docs/kv_profile_skill/03_profile_design | step3_profile_design.md | profiling 设计方案 |
| Step 4 | TODO | docs/kv_profile_skill/04_incremental_implementation | step4_patch_plan.md | 逐 patch 手动实现方案 |
```

Status can only be:

```text
TODO
IN_PROGRESS
DONE
BLOCKED
SKIPPED
```

---

## Final Report

After finishing Step 0, report in Chinese:

1. What folders were created.
2. What status file was created.
3. Confirm that no vLLM source code was modified.
4. Tell me that the next step should be Step 1.
5. Show `git status` for the lab repo only.

Do not execute Step 1.
