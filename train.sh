#train.sh

#!/bin/bash
# Pick which physical GPUs to use (e.g. "2,3" for the 3rd and 4th card)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# num_processes must match the count of GPUs listed above
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

accelerate launch \
	--mixed_precision bf16 \
	--main_process_port=29501 \
	--num_processes=$NUM_GPUS \
	train.py \
	--models '2toINF/X-VLA-Pt' \
	--train_metas_path /data2/daniel/libero/libero_h5/libero_meta.json \
	--learning_rate 1e-4 \
	--learning_coef 0.1 \
	--iters 50000 \
	--freeze_steps 1000 \
	--warmup_steps 2000 \
	--save_interval 1000 \
	--seed 142 \
	--batch_size 16
