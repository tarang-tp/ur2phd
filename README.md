# ur2phd

This project implements the SST-2 order-sensitivity experiment from the proposal:

- fine-tune `distilbert-base-uncased` multiple times on SST-2,
- keep initialization and hyperparameters fixed,
- vary only the order of training examples,
- export prediction logs for each run,
- analyze disagreement, confidence variance, and a lightweight stability score.

## What It Produces

Each training run writes:

- `artifacts/<run_name>/best_model/`
- `artifacts/<run_name>/metrics.json`
- `artifacts/<run_name>/train_predictions.csv`
- `artifacts/<run_name>/validation_predictions.csv`

The analysis step writes:

- `analysis/sentence_summary.csv`
- `analysis/pairwise_agreement.csv`
- `analysis/model_scores.csv`
- `analysis/stable_vs_unstable_comparison.csv`
- `analysis/top_unstable_sentences.csv`
- `analysis/summary.json`
- plots for disagreement, pairwise agreement, confidence variance, and agreement-vs-accuracy

## Setup

```bash
pip install -r requirements.txt
```

## Run The Full 10-Model Sweep

```bash
python run_experiment.py --runs 10 --base-permutation-seed 100 --model-seed 13
```

Optional flags:

- `--fp16` to reduce GPU memory usage
- `--batch-size 32`
- `--epochs 3`
- `--learning-rate 2e-5`

## Run A Single Training Job

```bash
python train_sst2.py --run-name run_00_perm_100 --permutation-seed 100 --model-seed 13
```

## Analyze Completed Runs

```bash
python analyze_runs.py --artifacts-root artifacts --output-dir analysis
```

## Notes On Experimental Control

- Model initialization is controlled by `--model-seed`.
- Data order is controlled by `--permutation-seed`.
- The training dataset is explicitly permuted before training.
- Trainer-side reshuffling is disabled so batch order stays identical to the chosen permutation.
- The validation split remains fixed across all runs.

## Main Metrics Implemented

- per-sentence disagreement count
- pairwise agreement matrix
- confidence variance over positive-class probability
- accuracy spread across runs
- majority-agreement stability score for model selection
- correlation between stability score and validation accuracy
- stable vs unstable sentence comparisons on:
  - token length
  - mean confidence
  - confidence variance
  - negation markers
  - contrast markers

## Expected Workflow

1. Run the 10 fine-tuning jobs.
2. Inspect `artifacts/*/metrics.json` to confirm accuracies are sensible.
3. Run `analyze_runs.py`.
4. Use `analysis/summary.json` and the generated plots in the writeup.

## Caveats

- Downloading SST-2 and DistilBERT requires internet access the first time.
- Exact bit-for-bit reproducibility can still depend on hardware and CUDA kernels.
- If you want stronger determinism on GPU, run on the same machine and software stack for all 10 jobs.
