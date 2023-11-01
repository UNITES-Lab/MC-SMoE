export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29510 mcsmoe/prune-non-core-experts-and-kd-downstream.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --preprocessing_num_workers=8 \
  --num_epochs=20 \
  --no_eval_until_epochs=1 \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --weight_decay=0.01 \
  --kd_temperature=2 \
  --mlm_lambda=1.0 \
  --kd_lambda=0.2 \
  --hd_lambda=0 \
  --task='copa' \
  --merging_strategy='drop' \
  --exact_fisher=False \
  --num_samples_for_merging=256 \
  --similarity_base='random' \
  --similarity_fn='cosine' \
  --num_groups=8 \
  --globally_group=True \
  --encoder_merging_layers="3,5,7,9,11" \
  --decoder_merging_layers="1,3,5,7,9,11" \
  --output_dir="results/copa/iterative-merging/" \
  --teacher_checkpoint="results/copa/vanilla-32e-acc0.68-permuted"