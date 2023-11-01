export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0


accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29510 mcsmoe/permute-model.py \
  --checkpoint="results/copa/vanilla-32e-acc0.68" \
  --save_dir="results/copa/vanilla-32e-acc0.68-permuted" \
  --include_wo=True
