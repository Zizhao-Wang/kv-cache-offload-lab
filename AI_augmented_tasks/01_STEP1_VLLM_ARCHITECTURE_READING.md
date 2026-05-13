# Step 1 Prompt: Beginner-Friendly vLLM Architecture Reading

You are my **vLLM / KV Cache Offloading Research Assistant**.

This is **Step 1** of the workflow.
Your job is to explain the overall vLLM architecture and request flow for a beginner.

Your final report must be written in **Chinese**.

---

## Hard Rules

1. Do **not** modify vLLM source code.
2. Do **not** implement profiling.
3. Do **not** create patches.
4. Do **not** execute Step 2/3/4.
5. Only read code and write a clear Markdown explanation.
6. The report must contain clickable Markdown links to source files.
7. The report must include diagrams or ASCII flow charts.
8. Be detailed and logical, but do not write a huge unreadable report.
9. This prompt is in English, but your output must be Chinese.

---

## Preconditions

Before starting, check that Step 0 has been done:

```bash
test -d /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill
cat /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/STATUS.md
```

If the folder or status file does not exist, stop and ask me to run Step 0 first.

---

## Workspace

vLLM source repo:

```bash
/home/jeff-wang/vllm_lab/repos/vllm
```

Output file:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/01_architecture/step1_vllm_architecture.md
```

When writing clickable source links from this output file to the vLLM repo, the relative path will usually look like:

```markdown
[`some_file.py`](../../../../vllm/vllm/.../some_file.py)
```

If you are not sure about relative links, include both:

1. a clickable relative Markdown link, and
2. the absolute path in plain text.

---

## Required Environment Checks

Run:

```bash
cd /home/jeff-wang/vllm_lab/repos/vllm
git status
git branch --show-current
git rev-parse --short HEAD
git remote -v
```

Then run:

```bash
source /home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/scripts/local_4090/activate_vllm_dev.sh
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("vllm:", vllm.__version__)
print("vllm file:", vllm.__file__)
PY
```

If any command fails, stop and explain in Chinese.

---

## Required Code Searches

Use `rg` to locate real code paths. Do not guess.

```bash
cd /home/jeff-wang/vllm_lab/repos/vllm

rg -n "class LLM|def generate|def chat" vllm
rg -n "class EngineCore|def add_request|def step" vllm
rg -n "class Scheduler|def schedule" vllm
rg -n "class GPUModelRunner|def execute_model|def _model_forward" vllm
rg -n "OpenAI|api_server|serve|entry_points" vllm
```

You may run extra `rg` commands if needed.

---

## Required Output Structure

Create the Markdown report:

```bash
/home/jeff-wang/vllm_lab/repos/kv-cache-offload-lab/docs/kv_profile_skill/01_architecture/step1_vllm_architecture.md
```

The report must be in Chinese and contain:

```markdown
# Step 1: vLLM 整体架构解读

## 0. 本步骤结论
A short summary: how to understand vLLM at a high level.

## 1. vLLM 有没有一个 main 函数？
Explain that vLLM does not have one single C-style main path for all use cases. Explain offline API, OpenAI server, engine, scheduler, worker/model runner.

## 2. vLLM 的核心心智模型
Use a simple diagram:

User request
  ↓
LLM API / OpenAI server
  ↓
Engine / EngineCore
  ↓
Scheduler
  ↓
KV cache management
  ↓
GPUModelRunner / model forward
  ↓
Output processing

Explain each box in beginner-friendly Chinese.

## 3. Offline Python API 请求路径
Explain how `LLM.generate()` or similar APIs enter vLLM.
Include clickable source links.

## 4. OpenAI Server 请求路径
Explain where server-style requests enter.
Include clickable source links.

## 5. Engine / EngineCore 做什么
Explain request admission and engine step logic.
Include clickable source links.

## 6. Scheduler 做什么
Explain what is scheduled: requests, tokens, budgets, running/waiting queues.
Do not deeply analyze KV cache yet; that is Step 2.
Include clickable source links.

## 7. Worker / GPUModelRunner 做什么
Explain where model execution happens.
Include clickable source links.

## 8. 从用户请求到输出的完整流程图
Use an ASCII diagram or Mermaid diagram plus short explanation.

## 9. 新手应该按什么顺序读代码
Give a reading order with file links, what to understand, and what to ignore.

## 10. 关键文件和函数表
Table columns:
- clickable source link
- class/function
- approximate line number from `rg -n`
- role
- why it matters
- what to ignore for now

## 11. 我如何自己复查这些结论
List the exact `rg` commands and what each command proves.

## 12. 本步骤没有覆盖什么
Clearly say that KV cache allocation/free/offload/recompute are left for Step 2.
```

---

## Clickable Link Requirement

For every important function, provide a clickable link like:

```markdown
[`Scheduler.schedule()`](../../../../vllm/vllm/v1/core/sched/scheduler.py) around line XXX
```

Do not only write absolute paths. I need to click the file in VSCode or Markdown preview.

---

## Completion Criteria

Step 1 is DONE only if:

1. The output file exists.
2. The report is in Chinese.
3. It includes architecture diagrams.
4. It includes clickable local Markdown links to code.
5. It explains the request input/output flow.
6. It does not modify vLLM source code.
7. It updates `STATUS.md`: Step 1 = DONE.
8. It tells me the next step is Step 2.
