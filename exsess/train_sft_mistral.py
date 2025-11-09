#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from typing import Dict, List

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# ----------------------------
# Chat template: inline or file
# ----------------------------

DEFAULT_CHAT_TEMPLATE_JSON = r'''{
  "chat_template": "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is \" + today + \".\\n\\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\\"What are some good restaurants around me?\\\" => \\\"Where are you?\\\" or \\\"When is the next flight to Tokyo\\\" => \\\"Where do you travel from?\\\")\" %}\n\n{{- bos_token }}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- if messages[0] is string %}\n        {%- set system_message = messages[0]['content'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- else %} \n        {%- set system_message = messages[0]['content'][0]['text'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- endif %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n{%- for message in loop_messages %}\n    {%- if message['role'] == 'user' %}\n            {%- if message['content'] is string %}\n            {{- '[INST]' + message['content'] + '[/INST]' }}\n            {%- else %}\n                    {{- '[INST]' }}\n                    {%- for block in message['content'] %}\n                            {%- if block['type'] == 'text' %}\n                                    {{- block['text'] }}\n                            {%- elif block['type'] == 'image' or block['type'] == 'image_url' %}\n                                    {{- '[IMG]' }}\n                                {%- else %}\n                                    {{- raise_exception('Only text and image blocks are supported in message content!') }}\n                                {%- endif %}\n                        {%- endfor %}\n                    {{- '[/INST]' }}\n                {%- endif %}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {%- if message['content'] is string %}\n            {{- message['content'] + eos_token }}\n        {%- else %}\n            {{- message['content'][0]['text'] + eos_token }}\n        {%- endif %}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}"
}'''

def load_chat_template(chat_template_json: str = None) -> str:
    """
    Returns the chat template string to store on the tokenizer.
    If chat_template_json is provided, it must be a JSON object with a 'chat_template' field.
    Otherwise uses DEFAULT_CHAT_TEMPLATE_JSON.
    """
    raw = chat_template_json if chat_template_json is not None else DEFAULT_CHAT_TEMPLATE_JSON
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Chat template JSON is invalid: {e}")
    if "chat_template" not in obj or not isinstance(obj["chat_template"], str):
        raise ValueError("Chat template JSON must include a 'chat_template' string field.")
    return obj["chat_template"]


# ----------------------------
# Data: JSONL reader + split
# ----------------------------

def read_jsonl(path: str) -> List[Dict]:
    allowed_tasks = {"translate_idiomatic", "transform", "contrast_judge"}
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}")

            # Require prompt/target fields to be present
            if "prompt" not in obj or "target" not in obj:
                raise ValueError(f"Line {i} missing 'prompt' or 'target' field.")

            # Keep only allowed tasks; skip if missing or not allowed
            task = obj.get("task")
            if task not in allowed_tasks:
                continue

            prompt = str(obj["prompt"]).strip()
            target = str(obj["target"]).strip()
            if prompt == "" or target == "":
                continue  # skip empty items

            items.append({
                "prompt": prompt,
                "target": target,
                "id": obj.get("id", f"item{i:06d}")
            })

    if not items:
        raise ValueError("No valid items found in JSONL.")
    return items

def to_messages(ex):
    return {
        "messages": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["target"]},
        ]
    }


# ----------------------------
# Main: Trainer + DeepSpeed
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/",
                        help="Local path or HF id for Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_file", type=str, default="out_sft/sft_items.jsonl",
                        help="Path to sft_items.jsonl")
    parser.add_argument("--output_dir", type=str, default='results/mistral_sft_full',
                        help="Where to save checkpoints and final model/tokenizer")
    parser.add_argument("--deepspeed", type=str, default='./deepspeed_config.json',
                        help="Path to DeepSpeed config JSON")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Provided by DeepSpeed/torchrun.")

    # New CLI options for chat template
    parser.add_argument("--chat_template_file", type=str, default=None,
                        help="Path to a JSON file containing {'chat_template': '...'}; overrides the inline default.")
    parser.add_argument("--force_chat_template", action="store_true", default=False,
                        help="If set, always set/save this chat template even if one already exists on the tokenizer.")

    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Load data
    all_items = read_jsonl(args.train_file)                    # list[dict]
    raw_ds   = Dataset.from_list(all_items)                    # HF Dataset
    ds = raw_ds.map(to_messages, remove_columns=raw_ds.column_names)
    split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[Data] Train: {len(train_ds)} | Eval: {len(eval_ds)} | Total: {len(ds)}")

    # ---- Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, local_files_only=args.local_files_only
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure BOS exists for the template (fallback to EOS if missing)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    # --- Inject chat template if needed/forced
    need_template = args.force_chat_template or not getattr(tokenizer, "chat_template", None)
    if need_template:
        if args.chat_template_file:
            with open(args.chat_template_file, "r", encoding="utf-8") as f:
                chat_template_str = load_chat_template(f.read())
        else:
            chat_template_str = load_chat_template()  # from DEFAULT_CHAT_TEMPLATE_JSON above

        tokenizer.chat_template = chat_template_str
        # Persist with the tokenizer so future loads pick it up automatically
        tokenizer.save_pretrained(args.output_dir)
        print("[ChatTemplate] Saved chat_template to tokenizer config in output_dir.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    # Gradient checkpointing + no cache
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False

    # ---- Build text-form datasets using the chat template (max compatibility)
    def chat_formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    text_train = train_ds.map(chat_formatting_func, batched=True, remove_columns=train_ds.column_names)
    text_eval  = eval_ds.map(chat_formatting_func,  batched=True, remove_columns=eval_ds.column_names)

    # ---- Training args (DeepSpeed integrated)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=not args.bf16,
        deepspeed=args.deepspeed,
        report_to=None if args.report_to == "none" else args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        logging_first_step=True,
        remove_unused_columns=False,            # IMPORTANT for causal LM + custom collator
        torch_compile=False,                    # keep False with DeepSpeed ZeRO
        label_smoothing_factor=args.label_smoothing,
        ddp_find_unused_parameters=False,
    )

    # -------- SFT Trainer --------
    print("Building SFTTrainer…")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            chat_template_path="../Latin-chatbot/models/mistral-7b-instruct-v0.3/chat_template.json",
            per_device_train_batch_size=args.per_device_train_batch_size,
            deepspeed=args.deepspeed,
        ),
        train_dataset=text_train,
        eval_dataset=text_eval
    )
    trainer.train()
    # ---- Quick sanity check for the template
    demo = {
        "messages": [
            {"role": "user", "content": "Say hi in Latin."},
            {"role": "assistant", "content": "Salve!"}
        ]
    }
    demo_text = tokenizer.apply_chat_template(demo["messages"], tokenize=False, add_generation_prompt=False)
    print("[ChatTemplate Demo]\n", demo_text[500:].replace("\n", "\\n"), "...\n")

    # ---- Train
    trainer.train()

    # ---- Save final
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Eval perplexity (quick)
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        try:
            ppl = math.exp(eval_loss)
        except OverflowError:
            ppl = float("inf")
        print(f"[Eval] loss={eval_loss:.4f} | ppl={ppl:.2f}")
        metrics["perplexity"] = ppl
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    print("Done.")


if __name__ == "__main__":
    main()
