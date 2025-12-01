# test_load_mistral3_cpu.py
import os, torch
from transformers import Mistral3ForConditionalGeneration, AutoProcessor  # << key change

MODEL_ID = "/reference/LLMs/Mistral_AI/Mistral-Small-3.1-24B-Instruct-2503-hf/"
DTYPE = torch.bfloat16   # use torch.float32 if bf16 on CPU complains

def main():
    print("Torch:", torch.__version__)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # tokenizer+image processor in one (works fine for text-only too)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    print("✅ Processor loaded")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,      # safe to keep for local repos
        torch_dtype=DTYPE,
        device_map={"": "cpu"},      # force CPU
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("✅ Model loaded")

    # tiny text-only smoke test (no images)
    inputs = processor(text="<s>[INST]Say hi![/INST]", return_tensors="pt")
    _ = model(**inputs)   # forward pass only (will be slow on CPU)
    print("✅ Forward pass ok")

if __name__ == "__main__":
    main()
    