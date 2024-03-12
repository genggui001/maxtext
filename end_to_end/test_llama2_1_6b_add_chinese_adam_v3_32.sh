#!/bin/bash
set -e

# export JAX_TRACEBACK_FILTERING=off

idx=$(date +%Y-%m-%d-%H-%M)
dataset_path=/home/genggui001/gdrive/gg-nlp-lm-new


all_token=2147483648000
all_batch_token=4194304
warmup_token=8388608000

max_target_length=4096
devices=32
per_device_batch_size=8

gradient_accumulation_steps=$(($all_batch_token / $max_target_length / $devices / $per_device_batch_size))
steps=$(($all_token / $max_target_length / $devices / $per_device_batch_size))
eval_interval=$(($steps / 1024))

echo "all_token=$all_token"
echo "all_batch_token=$all_batch_token"
echo "max_target_length=$max_target_length"
echo "devices=$devices"
echo "per_device_batch_size=$per_device_batch_size"
echo "gradient_accumulation_steps=$gradient_accumulation_steps"
echo "steps=$steps"
echo "eval_interval=$eval_interval"


# export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

# python3 -u MaxText/train.py MaxText/configs/base.yml \
#  run_name=runner_${idx} \
#  model_name='llama2-1_6b-add-chinese' \
#  ici_tensor_parallelism=4 \
#  steps=$steps \
#  warmup_steps_fraction=0 \
#  eval_interval=$eval_interval \
#  checkpoint_period=$eval_interval \
#  checkpoint_max_to_keep=5 \
#  checkpoint_save_best=True \
#  max_target_length=$max_target_length \
#  per_device_batch_size=$per_device_batch_size \
#  gradient_accumulation_steps=$gradient_accumulation_steps \
#  base_output_directory=/home/genggui001/code/maxtext/tmp/llama2-1_6b-add-chinese  \
#  dataset_path=${dataset_path} \
#  attention=dot_product \
#  opt_type=adamw \
#  adam_b1=0.9 \
#  adam_b2=0.95 \
#  adam_weight_decay=0.1
