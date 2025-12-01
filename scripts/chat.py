from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- config ---
model_name = "results/sweep_lr1e-06_bs2_ep1_nopack/checkpoint-80"  # local fine-tune
use_multi_gpu = True  # set False if you want only a single GPU

# --- load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- load model on GPU(s) ---
if use_multi_gpu and torch.cuda.device_count() > 1:
    # Shard across visible GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",        # <- uses all visible GPUs
        torch_dtype=torch.float16 # or torch.bfloat16 if your GPUs support it
    )
else:
    # Single GPU (or CPU fallback)
    print("CUDA devices:", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to(device)

# --- quick sanity print for placement ---
print("Device map:", getattr(model, "hf_device_map", "single-device"))
if not torch.cuda.is_available():
    print("WARNING: CUDA not available; running on CPU.")

# --- chat loop ---
messages = [
    {"role": "system", "content": """You are a helpfull assistant."""},
]

print("Ready. Type your message (Ctrl+C to exit).")
while True:
    try:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})

        # Render with chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize; send inputs to an appropriate device
        # For sharded models, sending to cuda:0 works well
        input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(input_device)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                do_sample=False,                 # greedy; flip to True + temperature/top_p if you want sampling
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ==== ONLY NEW TOKENS ====
        # Slice off the prompt length to keep *only* the model's continuation
        input_len = model_inputs["input_ids"].shape[-1]
        new_tokens = generated[0, input_len:]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print("Assistant:", assistant_text)

        # add assistant turn back to history
        messages.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print("\nBye!")
        break
