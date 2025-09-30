```
$ nohup bash -c 'CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct' > vllm.log 2>&1 &
$ nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config recipes/custom.yaml' > training.log 2>&1 &
```
