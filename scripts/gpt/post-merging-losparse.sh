export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29511 mcmoe/fsgpt-zero-shot-losparse.py \
  --checkpoint="results/fsgpt/openbookqa/merging/router-logits" \
  --output_dir="results/fsgpt/openbookqa" \
  --task="openbookqa" \
  --final_threshold=0.10 \
  --low_rank_factor=32 \
  --sample_batch_size=16 \
  --eval_batch_size=32 \
  --compression_layers="1,3,5,7,9,11"