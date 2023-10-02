export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=5

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29515 mcmoe/evaluate-fsgpt-merged-zero-shot.py \
  --checkpoint="results/fsgpt/mrpc/merging/router-logits/" \
  --task="mrpc" \
  --eval_batch_size=32