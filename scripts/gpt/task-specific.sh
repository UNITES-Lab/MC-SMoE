export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29510 mcsmoe/fsgpt-zero-shot-with-task-specific-pruning.py \
  --task="mrpc" \
  --minimum_num_experts=16 \
  --threshold_beta=0.1 \
  --num_pruning_times=5 \
  --pruning_layers="1,3,5,7,9,11" \
  --output_dir="results/fsgpt/task-specific" \
  --checkpoint="Phando/fairseq-moe-15b-bf16"