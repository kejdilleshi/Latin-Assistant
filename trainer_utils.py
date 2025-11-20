"""Trainer configuration utilities for SFT training."""

from trl import SFTTrainer, SFTConfig


def create_trainer(model, train_ds, val_ds, args):
    """Create and configure the SFTTrainer."""
    print("Building SFTTrainer…")

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
        load_best_model_at_end=True,  # Load best model at end of training
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
    )

    return trainer
