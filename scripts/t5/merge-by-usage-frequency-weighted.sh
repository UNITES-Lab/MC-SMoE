export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging.py \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --preprocessing_num_workers=8 \
  --num_epochs=10 \
  --no_eval_until_epochs=0 \
  --num_eval_steps=50 \
  --learning_rate=2e-4 \
  --warmup_steps=16 \
  --weight_decay=0.01 \
  --kd_temperature=2 \
  --mlm_lambda=1.0 \
  --kd_lambda=0.2 \
  --hd_lambda=0 \
  --task="copa" \
  --merging_strategy="normal" \
  --exact_fisher=False \
  --num_samples_for_merging=256 \
  --similarity_base="router-logits" \
  --reverse_similarity=False \
  --similarity_fn="cosine" \
  --num_groups=8 \
  --globally_group=True \
  --permute_when_merge=False \
  --save_stable_rank=False \
  --encoder_merging_layers="3,5,7,9,11" \
  --decoder_merging_layers="1,3,5,7,9,11" \
  --output_dir="results/copa/merging/" \
  --teacher_checkpoint="results/copa/vanilla-32e-acc0.68-permuted"