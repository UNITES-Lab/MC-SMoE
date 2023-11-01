export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/t5-finetune-on-downstream.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_epochs=20 \
  --save_each_epoch=False \
  --preprocessing_num_workers=8 \
  --task="copa" \
  --learning_rate=1e-4 \
  --warmup_steps=8 \
  --output_dir="results/copa/t5-bs8-lr-1e-4"