# Step 4 Prompt: Incremental Manual Implementation Guide

You are my **vLLM / KV Cache Offloading Research Coding Assistant**.

This is **Step 4** of the workflow.
Your job is **not** to implement everything automatically.
Your job is to provide an incremental, manual, patch-by-patch implementation guide based on Step 3.

Your final report must be written in **Chinese**.

---

## Hard Rules

1. Do **not** implement the whole profiling system at once.
2. Do **not** create a huge patch.
3. Do **not** modify CUDA kernels.
4. Do **not** change scheduler policy.
5. Do **not** change KV tensor layout.
6. If you modify code in a future patch task, modify only the single patch requested by the user.
7. In this Step 4 planning run, prefer to produce manual instructions first.
8. The output must be Chinese.
9. This prompt is in English, but your output must be Chinese.

---

## Preconditions

Step 1, Step 2, and Step 3 must be DONE.

Check:

```bash
cat /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/STATUS.md
test -f /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/03_profile_design/step3_profile_design.md
```

If Step 3 is missing, stop and ask me to finish Step 3 first.

---

## Output File

Create:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/04_incremental_implementation/step4_patch_plan.md
```

---

## Required Output Structure

The report must be in Chinese and contain:

```markdown
# Step 4: 逐 Patch 手动实现方案

## 0. 本步骤结论
Explain that implementation must be incremental.

## 1. 总体原则
Explain:
- one patch at a time
- each patch must be independently testable
- no huge all-in-one implementation
- no CUDA kernel changes now
- no scheduler policy changes now

## 2. Patch 0: baseline verification before code changes
Commands to verify current vLLM still works.
No source changes.

## 3. Patch 1: JSONL trace writer only
Include:
- target file
- allowed files
- forbidden files
- code logic to add
- example code snippet if useful
- verification command
- expected output
- revert method
- risk level

## 4. Patch 2: scheduler step logging only
Same structure.
Must depend on Patch 1.
Do not log KV blocks yet.

## 5. Patch 3: KV allocation logging only
Same structure.
Do not add free logging yet.

## 6. Patch 4: KV free logging only
Same structure.

## 7. Patch 5: prefix cache hit/miss logging only
Same structure.
Include repeated-prefix test idea.

## 8. Patch 6: preemption/recompute candidate logging only
Same structure.
Do not force preemption unless I explicitly ask.

## 9. Patch 7: model forward timing only
Same structure.
Explain CUDA async timing and when to use `torch.cuda.Event`.
Warn about CUDA graph risk.

## 10. Patch 8: trace analyzer only
Lab repo script only.
Do not modify vLLM source.

## 11. How I should review each patch
For each patch, explain:
- what file to open
- what function to inspect
- what event should appear in trace
- what failure means

## 12. How to stop safely
Explain how to revert a patch and return to clean state.
```

---

## Patch Description Requirements

For every patch, include a table:

```text
Patch ID
Goal
Target file
Target function/class
Allowed changes
Forbidden changes
Expected trace events
Verification command
Expected output
Risk level
Rollback command
```

---

## Optional Code Snippets

You may include small code snippets as examples, but do not ask to paste a large all-in-one diff.

Each snippet must be small and local.

The style should be:

```text
Open file X.
Find function Y.
Add this small block near Z.
Run this command.
Check this output.
```

---

## Completion Criteria

Step 4 is DONE only if:

1. The output file exists.
2. The report is in Chinese.
3. It gives patch-by-patch manual instructions.
4. It does not propose one huge patch.
5. It includes verification and rollback for every patch.
6. It updates `STATUS.md`: Step 4 = DONE for planning, not necessarily code implementation.
7. It clearly says which patch should be implemented first.
