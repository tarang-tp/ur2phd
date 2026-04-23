from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple SST-2 order-sensitivity training runs.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-permutation-seed", type=int, default=100)
    parser.add_argument("--model-seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
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
    args.output_root.mkdir(parents=True, exist_ok=True)

    for index in range(args.runs):
        permutation_seed = args.base_permutation_seed + index
        run_name = f"run_{index:02d}_perm_{permutation_seed}"
        command = [
            sys.executable,
            "train_sst2.py",
            "--run-name",
            run_name,
            "--permutation-seed",
            str(permutation_seed),
            "--model-seed",
            str(args.model_seed),
            "--output-root",
            str(args.output_root),
            "--model-name",
            args.model_name,
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--warmup-ratio",
            str(args.warmup_ratio),
            "--weight-decay",
            str(args.weight_decay),
        ]

        if args.fp16:
            command.append("--fp16")

        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
