#! /bin/bash
export CUDA_VISIBLE_DEVICES='0'
model_name_or_path=/mnt/data1/AIModel/ZhipuAI/chatglm3-6b-32k/

train_file=../data/train_message.json
validation_file=../data/train_message.json
ABS_PATH=../
output_dir="$ABS_PATH/saved_models/"
mkdir -p ${output_dir}

cutoff_len=1024

#FT
 torchrun --nproc_per_node 1 sft_train.py \
     --ddp_timeout 36000 \
     --model_name_or_path ${model_name_or_path} \
     --model_type glm3 \
     --deepspeed configs/deepspeed_config.json \
     --train_path ${train_file} \
     --eval_path ${validation_file} \
     --per_device_train_batch_size 2 \
     --per_device_eval_batch_size 2 \
     --gradient_accumulation_steps 4 \
     --num_train_epochs 2 \
     --model_max_length ${cutoff_len} \
     --save_strategy "steps" \
     --save_total_limit 3 \
     --learning_rate 8e-6 \
     --weight_decay 0.00001 \
     --warmup_ratio 0.05 \
     --lr_scheduler_type "cosine" \
     --logging_steps 10 \
     --evaluation_strategy "steps" \
     --torch_dtype "bfloat16" \
     --bf16 \
     --seed 1234 \
     --gradient_checkpointing \
     --overwrite_output_dir \
     --output_dir ${output_dir} \
     --use_lora \
     --lora_config model_configs/glm_lora_config.json \
     --report_to "tensorboard"
    # --use_flash_attention
    # --resume_from_checkpoint ...

