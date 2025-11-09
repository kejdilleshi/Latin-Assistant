from transformers import AutoModelForCausalLM, AutoTokenizer


from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "results/smol_masked_20/checkpoint-1200"  # or "HuggingFaceTB/SmolLM3-3B"
model_name = "results/SmolLM3_EDC_sft/checkpoint-200" 
# model_name = "Qwen/Qwen3-4B-Instruct-2507"


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, local_files_only=True
)

# optional: ensure pad_token_id so generate doesn't complain
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# conversation state (multi-turn)
messages_think = [
    {"role": "system", "content": """You are a Robot."""},
]

print("Ready. Type your message (Ctrl+C to exit).")
while True:
    try:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        # add user turn
        messages_think.append({"role": "user", "content": user_text})

        # render with chat template
        text = tokenizer.apply_chat_template(
            messages_think,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenize & move to device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # generate
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            # temperature=0.7,
            # top_p=0.5,              # keep <= 1.0
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
        )
        # decode only the newly generated tokens
        output_ids = generated_ids[0][len(model_inputs):] # print only the generated part 
        # output_ids = generated_ids[0][:] # print all 
        assistant_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print("Assistant:", assistant_text)

        # add assistant turn back to history
        messages_think.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print("\nBye!")
        break
