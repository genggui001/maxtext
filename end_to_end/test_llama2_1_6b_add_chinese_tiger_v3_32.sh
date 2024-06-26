#!/bin/bash
set -e

# export JAX_TRACEBACK_FILTERING=off

idx=$(date +%Y-%m-%d-%H-%M)

# base_ckpt_path=/home/genggui001/gdrive/genggui001/pretrain_weights/nlp/llama2-7b-maxtext-add-chinese-other/0/default
dataset_path=/home/genggui001/gdrive/gg-nlp-lm-new

all_token=536870912000
all_batch_token=1048576

max_target_length=4096
devices=32
per_device_batch_size=4

warmup_steps=200

forward_steps=$(($all_token / $max_target_length / $devices / $per_device_batch_size))
gradient_accumulation_steps=$(($all_batch_token / $max_target_length / $devices / $per_device_batch_size))
forward_warmup_steps=$(($gradient_accumulation_steps * $warmup_steps))

eval_interval=$(($forward_steps / 256))

echo "all_token=$all_token"
echo "all_batch_token=$all_batch_token"

echo "max_target_length=$max_target_length"
echo "devices=$devices"
echo "per_device_batch_size=$per_device_batch_size"

echo "warmup_steps=$warmup_steps"

echo "forward_steps=$forward_steps"
echo "forward_warmup_steps=$forward_warmup_steps"
echo "gradient_accumulation_steps=$gradient_accumulation_steps"

echo "eval_interval=$eval_interval"

echo "----------------------------------------runner_${idx}--------------------------------------------------"

export LIBTPU_INIT_ARGS="TPU_MEGACORE=MEGACORE_DENSE"

python3 -u MaxText/train.py MaxText/configs/base.yml \
 run_name=runner_${idx} \
 model_name='llama2-1_6b-add-chinese' \
 ici_data_parallelism=-1 \
 ici_fsdp_parallelism=8 \
 data_shuffle_seed=123456 \
 steps=$forward_steps \
 warmup_steps=$forward_warmup_steps \
 eval_interval=$eval_interval \
 checkpoint_period=$eval_interval \
 checkpoint_max_to_keep=5 \
 checkpoint_save_best=True \
 max_target_length=$max_target_length \
 per_device_batch_size=$per_device_batch_size \
 gradient_accumulation_steps=$gradient_accumulation_steps \
 base_output_directory=gs://genggui001-europe-west4/llm_checkpoints/llama2-1_6b-add-chinese_tiger  \
 dataset_path=${dataset_path} \
 attention=dot_product \
 norm_head_weight=True \
 gradient_clipping_threshold=0.0 \
 opt_type=tiger \
 tiger_b=0.965 \
 tiger_weight_decay=0.025 \
 adam_b1=0.9 \
 adam_b2=0.95 \
 adam_weight_decay=0.1 \
 learning_rate=2.5e-05 \
 adam_learning_rate_fraction=10.0
