
# nohup python main.py --task citeseer_1 --config fast_gpt3.5_different --build --launcher_save_path "LLMGraph/llms/launcher_info_citeseer_gpt35_different.json" >> launcher_info_cora_gpt35_different_main.log 2>&1
nohup python main.py --task citeseer_1 --config fast_gpt3.5_shuffle --build --launcher_save_path "LLMGraph/llms/launcher_info_citeseer_gpt35_shuffle.json" >> launcher_info_citeseer_gpt35_shuffle_main.log 2>&1

# nohup python main.py --task cora_1 --config fast_gpt3.5_shuffle --build --launcher_save_path "LLMGraph/llms/launcher_info_cora_gpt35_shuffle.json" >> launcher_info_cora_gpt35_shuffle_main.log 2>&1
# nohup python main.py --task cora_1 --config fast_gpt3.5_different --build --launcher_save_path "LLMGraph/llms/launcher_info_cora_gpt35_different.json" >> launcher_info_cora_gpt35_different_main.log 2>&1
# nohup python main.py --task cora_1 --config fast_gpt4-mini_shuffle --build --launcher_save_path "LLMGraph/llms/launcher_info_citeseer_gpt4_shuffle.json" >> launcher_info_citeseer_gpt4_shuffle_main.log 2>&1
# nohup python main.py --task cora_1 --config fast_gpt4-mini_different --build --launcher_save_path "LLMGraph/llms/launcher_info_citeseer_gpt4_different.json" >> launcher_info_citeseer_gpt4_different_main.log 2>&1
