"""Trainer configuration utilities for SFT training."""

from trl import SFTTrainer, SFTConfig


def create_trainer(model, train_ds, val_ds, args):
    """Create and configure the SFTTrainer."""
    print("Building SFTTrainer…")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        do_eval=True,
        eval_steps=50,
        eval_strategy="steps",
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        packing=True,
        completion_only_loss=True,
        max_length=1024
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    return trainer
