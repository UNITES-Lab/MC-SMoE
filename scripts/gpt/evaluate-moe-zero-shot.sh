export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29513 mcsmoe/evaluate-fsgpt-zero-shot.py \
  --checkpoint="results/fairseq-moe-15b-bf16-permuted" \
  --tasks="mrpc" \
  --eval_batch_size=32