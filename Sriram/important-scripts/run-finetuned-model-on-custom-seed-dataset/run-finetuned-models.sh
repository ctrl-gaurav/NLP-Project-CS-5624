#!/bin/sh

python -m llmthinkbench.cli --model_path "/home/sriramsrinivasan/SFT/finetuned-Qwen2.5-3B-Instruct/final_model" --tensor_parallel_size 2 --gpu_memory_utilization 0.95 --temperature 0.7 --top_p 0.9 --max_tokens 1024 --tasks sorting comparison even_count find_minimum mean --list_sizes 8 16 --range -100 100 --datapoints 20 --store_details --seed 42 --folds 3

python -m llmthinkbench.cli --model_path "/home/sriramsrinivasan/SFT/finetuned-Qwen2.5-1.5B-Instruct/final_model" --tensor_parallel_size 2 --gpu_memory_utilization 0.95 --temperature 0.7 --top_p 0.9 --max_tokens 1024 --tasks sorting comparison even_count find_minimum mean --list_sizes 8 16 --range -100 100 --datapoints 20 --store_details --seed 42 --folds 3