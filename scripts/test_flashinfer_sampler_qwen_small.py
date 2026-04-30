from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        disable_log_stats=True,
        enforce_eager=True,
    )

    outputs = llm.generate(
        ["用一句话解释什么是 KV cache offloading。"],
        SamplingParams(max_tokens=64, temperature=0.0),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
