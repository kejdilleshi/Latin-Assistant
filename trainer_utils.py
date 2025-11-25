"""Trainer configuration utilities for SFT training."""

from trl import SFTTrainer, SFTConfig


def create_trainer(model, tokenizer, train_ds, val_ds, args):
    """Create and configure the SFTTrainer."""
    print("Building SFTTrainer…")

    # Debug: Test tokenizer before creating trainer
    print("\n[DEBUG] Testing tokenizer with sample message...")
    test_msg = [
        {"role": "user", "content": "Test prompt"},
        {"role": "assistant", "content": "Test response"}
    ]
    test_result = tokenizer.apply_chat_template(
        test_msg,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    print(f"[DEBUG] Test tokenization successful: {len(test_result.get('input_ids', []))} tokens")
    print(f"[DEBUG] Has assistant_masks: {'assistant_masks' in test_result}")
    if 'assistant_masks' in test_result:
        print(f"[DEBUG] Assistant tokens found: {sum(test_result['assistant_masks'])} out of {len(test_result['assistant_masks'])}")
    print()

    # Determine wandb settings
    report_to = ["wandb"] if args.use_wandb else []
    run_name = args.wandb_run_name if hasattr(args, 'wandb_run_name') and args.wandb_run_name else None

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        do_eval=True,
        eval_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",  # Align save strategy with eval strategy
        load_best_model_at_end=False,  # Load best model at end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss as the metric
        greater_is_better=False,  # Lower loss is better
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        packing=args.packing,
        assistant_only_loss=True,
        max_length=2048,
        warmup_ratio= 0.03,
        max_grad_norm=1.0,
        # Weights & Biases configuration
        report_to=report_to,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # Pass the tokenizer explicitly
    )

    return trainer
