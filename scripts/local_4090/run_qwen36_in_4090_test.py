from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    model_path = "/home/jeff-wang/vllm_lab/models/Qwen3.6-27B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    messages = [
        {"role": "user", "content": "用一句话解释什么是 KV cache offloading。"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=2,
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_log_stats=True,
    )

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=64, temperature=0.0),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()