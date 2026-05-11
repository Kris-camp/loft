#!/usr/bin/env bash
set -euo pipefail

# Example: CoLA, LOFT skewgrad-free, DeBERTaV3-base.
# This is the GLUE setting validated on Gadi.

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
  --output_dir outputs/cola_skewgrad_free_r46_lr5e-4_seed42 \
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
  --loft_pr_init wg_skew \
  --loft_use_cayley_neumann True \
  --loft_num_cayley_neumann_terms 5
