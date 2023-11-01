export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/switch-finetune-with-task-specific-pruning.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_epochs=20 \
  --no_eval_until_epochs=1 \
  --save_each_epoch=False \
  --num_eval_steps=100 \
  --preprocessing_num_workers=8 \
  --task="copa" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --minimum_num_experts=8 \
  --threshold_beta=0.1 \
  --inspect_interval_steps=4 \
  --num_pruning_times=5 \
  --encoder_pruning_layers="3,5,7,9,11" \
  --decoder_pruning_layers="1,3,5,7,9,11" \
  --output_dir="results/copa/vanilla-32e-acc0.68-task-specific" \
  --checkpoint="results/copa/vanilla-32e-acc0.68"