export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29513 mcmoe/profile-fsgpt-flops.py \
  --task="squad" \
  --checkpoint="Phando/fairseq-moe-15b-bf16" \
  --batch_size=64


