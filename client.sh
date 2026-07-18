conda activate libero
cd /data2/daniel/Documents/X-VLA-L2P/evaluation/libero

export CUDA_VISIBLE_DEVICES=1 # sim renders on a different GPU than the server
MUJOCO_GL=egl python libero_client.py \
	--server_ip 127.0.0.1 \
	--server_port 8010 \
	--output_dir /data2/daniel/libero/eval_out \
	--task_suites libero_10 \
	--eval_time 10 \
	--act_type abs
