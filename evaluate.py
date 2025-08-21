import os
import shutil

import yaml
import time
import openai
import json
def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list



def evaluate_tasks(
                   task_name, 
                   configs,
                   args,
              log_dir = "LLMGraph/tasks/cache",
              ):
    os.putenv("PYTHONPATH","/mnt2/jijiarui/LLM4Graph")
    
    success_configs = []
    failed_configs = []
    
    command_template = "python evaluate/article/main.py --task {task} --config {config} "+args
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_task_dir = os.path.join(log_dir,"log")
    if not os.path.exists(log_task_dir):
        os.makedirs(log_task_dir)

    complete_path = os.path.join(log_dir,"complete.json")            
    
    for idx, config in enumerate(configs):
        command = command_template.format(config = config,
                                        task = task_name,
                                        )
        print(command)
        try:
            success_configs.append(config.replace("\(","(").replace("\)",")"))
        except Exception as e:
            print(e)
            failed_configs.append(config.replace("\(","(").replace("\)",")"))

        with open(complete_path,'w',encoding = 'utf-8') as f:
            uncomplete_configs = configs[idx+1:] if (idx+1)< len(configs) else []
            json.dump({"success":success_configs,
                    "failed":failed_configs,
                    "uncomplete":uncomplete_configs}, 
                    f,
                    indent=4,
                    separators=(',', ':'),ensure_ascii=False)


def run_evaluation():
    llms = ["gpt3.5","gpt4-mini","vllm"]

    llm_agent_config_templates = [
       
        "template_search_shuffle_base_{llm}_powerlaw_base",
        "template_search_shuffle_base_{llm}_recall100f",
        "template_search_shuffle_base_{llm}_nocitetime",

        "template_search_shuffle_anonymous_{llm}",
        "template_search_shuffle_base_{llm}",
        "template_search_shuffle_base_nosocial_{llm}",
        "template_search_shuffle_equal_country_{llm}"
        ]
    
    task_name_map = {}
    task_name_map["llm_agent"] = []
    for llm in llms:
        task_name_map["llm_agent"].extend([config_template.format(llm = llm) 
                    for config_template in llm_agent_config_templates])
    task_name_map["llm_agent"].extend(["template_2engine"])

    task_name_map.update({
        "citeseer":["template_fast_{llm}".format(llm = llm) 
                        for llm in llms],
        "cora":["template_fast_{llm}".format(llm = llm) 
                        for llm in llms],
    })

    task_name_args_map = {
        # "llm_agent":["--threshold 500"],
        # "cora":["--threshold 5000"],
        "citeseer":["--threshold 5000"],
    }

    # debug
    task_name_map = {
        "cora_1": [
            "fast_gpt3.5",
            "fast_gpt3.5_different",
            "fast_gpt4-mini_different",
            "fast_gpt4-mini",
            "fast_vllm",
            "fast_llama3_different"
        ]
    }
    task_name_args_map = {
        # "llm_agent":["--threshold 500"],
        # "cora":["--threshold 5000"],
        "cora_1":["--threshold 5000"],
    }
    
    for idx, task_name in enumerate(task_name_map.keys()):        
        
        log_dir = f"LLMGraph/tasks/{task_name}/cache"
        configs = task_name_map[task_name]
        args = " ".join(task_name_args_map[task_name])
        evaluate_tasks( task_name,
                        configs=configs,
                        args = args,
                        log_dir = log_dir)
       
            
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# # 定义要运行的命令列表
commands = [
    # "python evaluate/article/main.py --task citeseer_1 --config fast_gpt3.5 --threshold 5000",
    # "python evaluate/article/main.py --task citeseer_1 --config fast_gpt3.5_different --threshold 5000",
    # "python evaluate/article/main.py --task citeseer_1 --config fast_gpt4-mini_different --threshold 5000",
    # "python evaluate/article/main.py --task citeseer_1 --config fast_gpt4-mini --threshold 5000",
    # "python evaluate/article/main.py --task citeseer_1 --config fast_vllm --threshold 5000",
    # "python evaluate/article/main.py --task citeseer_1 --config fast_llama3_different --threshold 5000",

    # "python evaluate/article/main.py --task cora_1 --config fast_gpt3.5 --threshold 5000",
    # "python evaluate/article/main.py --task cora_1 --config fast_gpt3.5_different --threshold 5000",
    # "python evaluate/article/main.py --task cora_1 --config fast_gpt4-mini_different --threshold 5000",
    # "python evaluate/article/main.py --task cora_1 --config fast_gpt4-mini --threshold 5000",
    # "python evaluate/article/main.py --task cora_1 --config fast_vllm --threshold 5000",
    # "python evaluate/article/main.py --task cora_1 --config fast_llama3_different --threshold 5000",

    "python evaluate/article/main.py --task llm_agent_1 --config search_shuffle_base_gpt3.5_powerlaw_different --threshold 500",
    "python evaluate/article/main.py --task llm_agent_1 --config search_shuffle_base_gpt4-mini_powerlaw_different --threshold 500",
    "python evaluate/article/main.py --task llm_agent_1 --config search_shuffle_base_vllm_powerlaw_different --threshold 500",
]

def run_command(cmd):
    """执行单个命令并输出结果"""
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return cmd, process.returncode

if __name__ == "__main__":
    max_workers = 4  # 根据你的CPU核心数调整
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        for future in as_completed(futures):
            cmd, retcode = future.result()
            print(f"Finished: {cmd} with return code {retcode}")



# if __name__ == "__main__":
#     run_evaluation()