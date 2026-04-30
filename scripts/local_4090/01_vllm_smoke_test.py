import os
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except Exception as e:
    print("[ERROR] Failed to import vllm.")
    print("Reason:", repr(e))
    sys.exit(1)

model_path = os.environ.get("MODEL_PATH", "").strip()

if not model_path:
    print("[ERROR] MODEL_PATH is empty.")
    print("Please set MODEL_PATH to your local model directory.")
    print("Example:")
    print("  export MODEL_PATH=/home/your_name/mlsys_lab/models/Qwen2.5-7B-Instruct")
    sys.exit(1)

model_dir = Path(model_path)

if not model_dir.exists():
    print(f"[ERROR] MODEL_PATH does not exist: {model_path}")
    sys.exit(1)

print("========== vLLM Smoke Test ==========")
print("MODEL_PATH:", model_path)

max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

print("MAX_MODEL_LEN:", max_model_len)
print("GPU_MEMORY_UTILIZATION:", gpu_memory_utilization)
print("TENSOR_PARALLEL_SIZE:", tensor_parallel_size)

print()
print("Loading model...")

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype="auto",
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    tensor_parallel_size=tensor_parallel_size,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

prompts = [
    "Explain KV cache in one simple sentence.",
]

print()
print("Generating...")

outputs = llm.generate(prompts, sampling_params)

print()
print("========== Output ==========")
for output in outputs:
    print("Prompt:")
    print(output.prompt)
    print()
    print("Generated text:")
    print(output.outputs[0].text)

print()
print("[OK] vLLM smoke test finished successfully.")
