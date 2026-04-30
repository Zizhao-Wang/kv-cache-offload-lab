from vllm import LLM, SamplingParams


def main():
    model_path = "/home/jeff-wang/vllm_lab/models/gemma-4-E4B-it"

    # Gemma instruction/chat style prompt.
    # 不要只给一句裸 prompt，先用明确的 chat 格式。
    prompt = (
        "<start_of_turn>user\n"
        "Explain KV cache offloading in one short paragraph. "
        "Do not answer with an empty response.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.7,
        top_p=0.95,
        stop=["<end_of_turn>"],
    )

    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        print("=" * 80)
        print("[PROMPT]")
        print(prompt)
        print("=" * 80)

        generated = output.outputs[0].text
        print("[RAW GENERATED repr]")
        print(repr(generated))

        print("=" * 80)
        print("[GENERATED TEXT]")
        print(generated)

        print("=" * 80)
        print("[DEBUG]")
        print("finish_reason:", output.outputs[0].finish_reason)
        print("num_output_tokens:", len(output.outputs[0].token_ids))


if __name__ == "__main__":
    main()