export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29513 mcmoe/permute-fsgpt-model.py \
  --checkpoint="Phando/fairseq-moe-15b-bf16" \
  --save_dir="results/fairseq-moe-15b-bf16-permuted"