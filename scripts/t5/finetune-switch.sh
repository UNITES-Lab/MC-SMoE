export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29513 mcsmoe/finetune-switch-transformers.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_epochs=20 \
  --no_eval_until_epochs=1 \
  --save_each_epoch=False \
  --preprocessing_num_workers=8 \
  --num_experts=32 \
  --task="copa" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --output_dir="results/copa/vanilla-32e-bs8-lr-1e-5"