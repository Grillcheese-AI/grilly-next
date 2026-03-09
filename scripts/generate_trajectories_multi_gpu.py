import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set. Please export it before running.")

    print(f"Loading {MODEL_ID} across 2x H200s via vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)

    # --- 1. Load and Merge All 5 Datasets (Optimized with Streaming) ---
    raw_prompts = []

    def extract_text_from_row(row):
        for col in ["instruction", "prompt", "question", "text", "content", "messages", "conversations"]:
            if col in row:
                return row[col]
        return list(row.values())[0]

    print("Streaming Pillar 1: validia-ai/distillery (Taking all ~50k)...")
    ds_distillery = load_dataset("validia-ai/distillery", split="train", streaming=True, token=HF_TOKEN)
    for row in ds_distillery: raw_prompts.append(extract_text_from_row(row))

    print("Streaming Pillar 2: xiao-hao/self-distillation-LLMs (Taking 100k subset)...")
    ds_self_distill = load_dataset("xiao-hao/self-distillation-LLMs", split="train", streaming=True, token=HF_TOKEN)
    for row in ds_self_distill.take(100000): raw_prompts.append(extract_text_from_row(row))

    print("Streaming Pillar 3: crownelius/Qwen3-Coder-Next-25k (Taking all 25k)...")
    ds_coder = load_dataset("crownelius/Qwen3-Coder-Next-25k", split="train", streaming=True, token=HF_TOKEN)
    for row in ds_coder: raw_prompts.append(extract_text_from_row(row))

    print("Streaming Pillar 4: nvidia/Nemotron-CC-v2 [Translated-Diverse-QA] (Taking 50k subset)...")
    ds_multilingual = load_dataset("nvidia/Nemotron-CC-v2", name="Translated-Diverse-QA", split="train", streaming=True, token=HF_TOKEN)
    for row in ds_multilingual.take(50000): raw_prompts.append(extract_text_from_row(row))

    print("Streaming Pillar 5: nvidia/Nemotron-Math-v2 [high_part00] (Taking 50k subset)...")
    ds_math = load_dataset("nvidia/Nemotron-Math-v2", split="high_part00", streaming=True, token=HF_TOKEN)
    for row in ds_math.take(50000): raw_prompts.append(extract_text_from_row(row))

    print(f"Total combined raw prompts loaded: {len(raw_prompts)}")

    # --- 2. Format with Qwen's Chat Template & FILTER LENGTH ---
    print("Applying Qwen chat template to all prompts...")
    formatted_prompts = []
    for raw_text in raw_prompts: 
        if not raw_text:
            continue
            
        if isinstance(raw_text, list) and len(raw_text) > 0 and isinstance(raw_text[0], dict):
            user_content = next((msg.get("content", "") for msg in raw_text if msg.get("role") == "user"), None)
            if not user_content:
                continue
            raw_text = user_content
            
        if not isinstance(raw_text, str):
            continue
            
        # FIX: Hard filter to drop insanely long prompts that break the 4096 VRAM limit
        # 12,000 chars is roughly 3,000 to 3,500 tokens. Well within safe limits.
        if len(raw_text) > 12000:
            continue
            
        conversation = [
            {"role": "system", "content": "You are a highly intelligent, multilingual reasoning engine and master programmer. Provide clear, direct, and optimal answers. For math, solve step-by-step."},
            {"role": "user", "content": raw_text}
        ]
        formatted = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    print(f"Successfully formatted {len(formatted_prompts)} prompts. Initializing vLLM...")

    # --- 3. Initialize vLLM (2x H200 configuration) ---
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=2, 
        dtype="bfloat16", 
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=250,
        repetition_penalty=1.05
    )

    # --- 4. Chunked Generation & Incremental Saving ---
    CHUNK_SIZE = 25000
    RESUME_INDEX = 150000  # <--- RESUMING EXACTLY WHERE IT CRASHED
    output_file = "teacher_golden_trajectories.jsonl"

    print(f"Blasting prompts through vLLM in chunks of {CHUNK_SIZE} (Resuming from {RESUME_INDEX})...")

    with open(output_file, 'a') as f:
        for i in range(RESUME_INDEX, len(formatted_prompts), CHUNK_SIZE):
            chunk = formatted_prompts[i : i + CHUNK_SIZE]
            print(f"Processing chunk {i} to {i + len(chunk)}...")
            
            outputs = llm.generate(chunk, sampling_params)
            
            for output in outputs:
                tokens = list(output.outputs[0].token_ids)
                if tokens:
                    f.write(json.dumps({"tokens": tokens}) + "\n")
                    
            f.flush()
            os.fsync(f.fileno())

    print(f"Successfully generated all trajectories to {output_file}")

if __name__ == "__main__":
    main()