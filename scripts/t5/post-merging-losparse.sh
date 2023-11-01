export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29510 mcsmoe/losparse-downstream.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --preprocessing_num_workers=8 \
  --num_epochs=50 \
  --no_eval_until_epochs=0 \
  --num_eval_steps=100 \
  --learning_rate=3e-5 \
  --warmup_steps=50 \
  --weight_decay=0.01 \
  --kd_temperature=2 \
  --mlm_lambda=1.0 \
  --kd_lambda=0.2 \
  --hd_lambda=0.0 \
  --task="copa" \
  --output_dir="results/copa/post-merging-losparse/" \
  --teacher_checkpoint="results/copa/vanilla-32e-acc0.68-permuted" \
  --student_checkpoint="results/copa/vanilla-32e-merged-8e" \
  --final_threshold=0.10 \
  --low_rank_factor=32
