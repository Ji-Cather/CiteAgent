import os
import shutil
import json
import yaml
import time
import openai

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list



def evaluate_tasks(
                   task_name, 
                   configs,
                   args,
              log_dir,
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
        "llm_agent":["--threshold 500"],
        "cora":["--threshold 5000"],
        "citeseer":["--threshold 10000"],
    }
    log_dir = f"LLMGraph/tasks/{task_name}/evaluate_cache"

    
    for idx, task_name in enumerate(task_name_map.keys()):        
        
        log_dir = f"LLMGraph/tasks/{task_name}/cache"
        configs = task_name_map[task_name]
        args = " ".join(task_name_args_map[task_name])
        evaluate_tasks( task_name,
                        configs=configs,
                        args = args,
                        log_dir = log_dir)
       
            



if __name__ == "__main__":
    run_evaluation()