from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import EvalPrediction, Trainer, set_seed
from torch.utils.data import SequentialSampler


NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "without",
    "isn't",
    "wasn't",
    "don't",
    "didn't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
}

CONTRAST_WORDS = {
    "but",
    "however",
    "although",
    "though",
    "yet",
    "despite",
    "unless",
    "except",
    "whereas",
    "while",
}


class NoShuffleTrainer(Trainer):
    """Preserve dataset order exactly as provided."""

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        return SequentialSampler(self.train_dataset)


@dataclass
class RunConfig:
    run_name: str
    permutation_seed: int
    model_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    max_length: int
    warmup_ratio: float
    weight_decay: float


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_sst2_dataset() -> dict[str, Dataset]:
    dataset = load_dataset("glue", "sst2")

    for split in ("train", "validation"):
        dataset[split] = dataset[split].add_column("example_id", list(range(len(dataset[split]))))

    return {"train": dataset["train"], "validation": dataset["validation"]}


def permute_training_split(train_dataset: Dataset, permutation_seed: int) -> Dataset:
    rng = np.random.default_rng(permutation_seed)
    indices = np.arange(len(train_dataset))
    rng.shuffle(indices)
    return train_dataset.select(indices.tolist())


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, desc="Tokenizing")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=True,
    )
    return tokenized


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def predictions_to_frame(
    *,
    run_name: str,
    split_name: str,
    raw_dataset: Dataset,
    logits: np.ndarray,
) -> pd.DataFrame:
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    positive_probs = probabilities[:, 1]

    frame = pd.DataFrame(
        {
            "run_name": run_name,
            "split": split_name,
            "example_id": raw_dataset["example_id"],
            "sentence": raw_dataset["sentence"],
            "label": raw_dataset["label"],
            "prediction": predictions,
            "prob_negative": probabilities[:, 0],
            "prob_positive": positive_probs,
            "confidence": probabilities.max(axis=1),
            "correct": predictions == np.asarray(raw_dataset["label"]),
        }
    )
    return frame


def token_length(text: str) -> int:
    return len(text.split())


def has_marker(text: str, lexicon: Iterable[str]) -> bool:
    lowered = f" {text.lower()} "
    return any(f" {token} " in lowered for token in lexicon)


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["token_length"] = enriched["sentence"].map(token_length)
    enriched["has_negation"] = enriched["sentence"].map(lambda text: has_marker(text, NEGATION_WORDS))
    enriched["has_contrast"] = enriched["sentence"].map(lambda text: has_marker(text, CONTRAST_WORDS))
    return enriched


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_metadata(config: RunConfig, metrics: dict[str, float]) -> dict:
    payload = asdict(config)
    payload["metrics"] = metrics
    return payload
