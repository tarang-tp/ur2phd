from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on SST-2 with a fixed training order.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=13)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
    )

    from ur2phd.experiment import (
        NoShuffleTrainer,
        RunConfig,
        compute_metrics,
        load_sst2_dataset,
        permute_training_split,
        predictions_to_frame,
        run_metadata,
        set_global_determinism,
        tokenize_dataset,
        write_json,
    )
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        run_name=args.run_name,
        permutation_seed=args.permutation_seed,
        model_seed=args.model_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    set_global_determinism(args.model_seed)
    raw_dataset = load_sst2_dataset()
    ordered_train = permute_training_split(raw_dataset["train"], args.permutation_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    tokenized_train = tokenize_dataset(ordered_train, tokenizer, args.max_length)
    tokenized_validation = tokenize_dataset(raw_dataset["validation"], tokenizer, args.max_length)

    training_args_kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.model_seed,
        "data_seed": args.model_seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": [],
        "fp16": args.fp16,
        "save_total_limit": 1,
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    validation_metrics = trainer.evaluate()
    train_predictions = trainer.predict(tokenized_train)
    validation_predictions = trainer.predict(tokenized_validation)

    train_frame = predictions_to_frame(
        run_name=args.run_name,
        split_name="train",
        raw_dataset=ordered_train,
        logits=train_predictions.predictions,
    )
    validation_frame = predictions_to_frame(
        run_name=args.run_name,
        split_name="validation",
        raw_dataset=raw_dataset["validation"],
        logits=validation_predictions.predictions,
    )

    train_frame.to_csv(run_dir / "train_predictions.csv", index=False)
    validation_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", run_metadata(config, validation_metrics))
    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))


if __name__ == "__main__":
    main()
