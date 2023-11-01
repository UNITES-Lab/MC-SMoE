export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29511 mcsmoe/merge-fsgpt-by-usage-frequency-weighting.py \
  --task="openbookqa" \
  --num_samples_for_merging=256 \
  --samples_batch_size=32 \
  --eval_batch_size=32 \
  --similarity_base="router-logits" \
  --similarity_fn="cosine" \
  --num_groups=16 \
  --merging_layers="1,3,5,7,9,11" \
  --output_dir="results/fsgpt/openbookqa/merging/" \
  --checkpoint="results/fairseq-moe-15b-bf16-permuted" \
  --debug=False