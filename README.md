# LOFT Code Release

This repository contains the code release for LOFT. The current initial release includes the GLUE experiments with DeBERTaV3-base, and additional experiment components may be added in future updates.

The repository currently keeps the GLUE training entry point and the LOFT components required by the GLUE experiments. Other experiment code can be added later under the same repository.

## Contents

- run_glue.py: GLUE / encoder-side training entry point.
- baselines/loft/: LOFT layer implementation.
- scripts/example_glue.sh: example GLUE launch command.
- requirements.txt: Python dependencies for GLUE experiments.

## Environment

The code was tested on Gadi with:

- Python 3.12
- PyTorch 2.2.0 + CUDA 11.8
- transformers 4.45.1
- datasets 4.5.0
- evaluate 0.4.6

On Gadi, the tested virtual environment was:

source /scratch/ca63/$USER/venvs/loft/bin/activate

For a fresh environment, install PyTorch for the target CUDA version first, then run:

pip install -r requirements.txt

## Data and model cache

The Gadi runs used local HuggingFace caches:

export HF_HOME=/scratch/ca63/$USER/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

For an environment with internet access, unset the offline flags and allow HuggingFace to download microsoft/deberta-v3-base and GLUE automatically.

## GLUE setup

We use DeBERTaV3-base as the encoder backbone.

Shared settings:

- Optimizer: AdamW
- Warmup ratio: 0.1
- LR schedule: linear
- Classifier-head LR: 5e-4
- Batch size: 32

Task-specific settings:

| Task | Max seq. len. | Epochs | Metric |
|---|---:|---:|---|
| CoLA | 64 | 20 | Matthews correlation |
| STS-B | 128 | 20 | Pearson correlation |
| MRPC | 256 | 30 | Accuracy |
| RTE | 256 | 30 | Accuracy |
| SST-2 | 128 | 10 | Accuracy |
| QNLI | 256 | 5 | Accuracy |

## LOFT variants

Main LOFT flags:

| Variant | Flags |
|---|---|
| Principal support | --loft_pr_init principal |
| Random support | --loft_pr_init random |
| Orthogonal transform | --loft_ortho True |
| Free transform | --loft_ortho False |

## Example CoLA command

The following is an example CoLA LOFT command:

python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name cola \
  --do_train True \
  --do_eval True \
  --max_seq_length 64 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-4 \
  --cls_learning_rate 5e-4 \
  --num_train_epochs 20 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --output_dir outputs/cola_loft_principal_r46_lr5e-4_seed42 \
  --overwrite_output_dir True \
  --logging_strategy epoch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_matthews_correlation \
  --greater_is_better True \
  --seed 42 \
  --data_fraction 1.0 \
  --subset_seed 42 \
  --peft_name loft \
  --peft_rank 46 \
  --loft_ortho False \
  --loft_pr_init principal \
  --loft_use_cayley_neumann True \
  --loft_num_cayley_neumann_terms 5

For a short smoke test, add:

--max_steps 20 --num_train_epochs 1 --evaluation_strategy no --save_strategy no

## Notes

This is a compact initial LOFT code release. Do not commit model weights, checkpoints, W&B logs, HuggingFace caches, or experiment outputs.
