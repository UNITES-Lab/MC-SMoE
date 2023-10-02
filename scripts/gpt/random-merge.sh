export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=5



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29515 mcmoe/random-merge-fsgpt.py \
  --task="mrpc" \
  --strategy="average" \
  --num_samples_for_merging=512 \
  --samples_batch_size=32 \
  --eval_batch_size=32 \
  --num_groups=16 \
  --merging_layers="1,3,5,7,9,11" \
  --output_dir="results/fsgpt/mrpc/random-merging/" \
  --checkpoint="Phando/fairseq-moe-15b-bf16" \
  --debug=False