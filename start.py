import os
import shutil
import json
import yaml
import time
import multiprocessing
from LLMGraph.utils.io import readinfo,writeinfo

import argparse
parser = argparse.ArgumentParser(description='experiment_runner')  # 创建解析器
parser.add_argument('--start_server', 
                    action='store_true',
                    default=False,
                    help="start server")

args = parser.parse_args()  # 解析参数


def start_launchers(launcher_num:int =8,
                    launcher_save_paths = [
                  "LLMGraph/llms/launcher_info.json"
              ]):
    command_template = "python start_launchers.py --launcher_num {launcher_num} --launcher_save_path {launcher_save_path}"
    
    # 创建多个进程，每个进程执行函数一次，并传入不同的参数
    processes = []
    for launcher_save_path in launcher_save_paths:
        command = command_template.format(launcher_num = launcher_num,
                                            launcher_save_path = launcher_save_path)

        p = multiprocessing.Process(target=os.system, 
                                    args=(command,))
        processes.append(p)
        p.start()
    
    # 等待所有进程执行结束
    for p in processes:
        p.join()

def run_tasks(configs,
              task_name,
              log_dir,
              launcher_save_paths = [
                  "LLMGraph/llms/launcher_info.json"
              ]):

    assert len(configs)==len(launcher_save_paths), "len not equal for launcher_save_paths"
    
    command_template = "python main.py --task {task} --config {config} --build --launcher_save_path {launcher_save_path}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_task_dir = os.path.join(log_dir,"log")
    if not os.path.exists(log_task_dir):
        os.makedirs(log_task_dir)

    complete_path = os.path.join(log_dir,"complete.json")            
    
    # 创建多个进程，每个进程执行函数一次，并传入不同的参数
    processes = []
    for idx, command_info in enumerate(zip(configs,launcher_save_paths)):
        config,launcher_save_path = command_info
        # launcher_save_path = "LLMGraph/llms/launcher_info_none.json"

        #test_article_num >= 1000
        path = f"LLMGraph/tasks/{task_name}/configs/{config}/data/log_info.json"
        if os.path.exists(path):
            log_info = readinfo(path)
            if log_info["generated_articles"] >= 1000-89:
                print("finished",config)
                continue

        command = command_template.format(config = config,
                                        task = task_name,
                                        launcher_save_path = launcher_save_path
                                        )
        p = multiprocessing.Process(target=os.system, 
                                    args=(command,))
        processes.append(p)
        p.start()

    # 等待所有进程执行结束
    success_configs = []
    failed_configs = []
    for config,p in zip(configs,processes):
        p.join()
        if p.exitcode == 0:
            success_configs.append(config)
        else:
            failed_configs.append(config)
    with open(complete_path,'w',encoding = 'utf-8') as f:
        json.dump({"success":success_configs,
                "failed":failed_configs}, 
                f,
                indent=4,
                separators=(',', ':'),ensure_ascii=False)
    

   


def run_experiments():
   
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
    
            
    run_simulation(task_name_map)
   


def run_simulation(
        task_name_map:dict # {task_name: List[config_name]}
        ):
    """ run experiments 
    Args:
        task_names (list): 
        configs (list): _description_
    """

    """
        We have also tested CiteAgent with "llama8b" "qwen2" "gemini-1.5-flash" and "mixtral". However, the human role-play capabilities of these LLMs fall short for simulating academic activities. We're planning to create a benchmark to evaluate the performance of these LLMs in human role-play scenarios, specifically from a social science perspective.
    """
    
    
    prefix = 0
    launcher_save_paths = []
    for task_name in task_name_map.keys():
        
        server_num = len(task_name_map[task_name])
        launcher_save_paths.extend([f"LLMGraph/llms/launcher_filter_{i}.json" for i in range(prefix+1,prefix+server_num+1)])
        prefix += server_num

    if args.start_server:    
        start_launchers(10,launcher_save_paths)
    else:
        for idx, task_name in enumerate(task_name_map.keys()):        
            server_num = len(task_name_map[task_name])
            launcher_save_paths_group = launcher_save_paths[idx*server_num:(idx+1)*server_num]
            log_dir = f"LLMGraph/tasks/{task_name}/cache"
            configs = task_name_map[task_name]
            run_tasks(configs,
                    task_name,
                    log_dir,
                    launcher_save_paths=launcher_save_paths_group)
            
    


def clear_experiment_cache(
                    configs,
                    task_name,):
    task_root = "LLMGraph/tasks"
    task_root = os.path.join(task_root,task_name)
    file_names =[
        "article_meta_info.pt",
        "author.pt"
    ]
  

    for config in configs:
        config_path = os.path.join(task_root,"configs",config)
        if not os.path.exists(config_path):
            print(config_path)
            continue
        data_dst = os.path.join(config_path,"data")
        data_src = os.path.join(task_root,"data")
        config_file = yaml.safe_load(open(os.path.join(config_path,"config.yaml")))
        # config_file["environment"]["article_write_configs"]["use_graph_deg"] = True
        with open(os.path.join(config_path,"config.yaml"), 'w') as outfile:
            yaml.dump(config_file, outfile, default_flow_style=False)

        if os.path.exists(data_dst):
            shutil.rmtree(data_dst)
        if os.path.exists(os.path.join(config_path,"evaluate")):
            shutil.rmtree(os.path.join(config_path,"evaluate"))
        os.makedirs(data_dst)
        for file_name in file_names:
            shutil.copyfile(os.path.join(data_src,file_name),
                            os.path.join(data_dst,file_name))
    
def modify_config_name_info(task_name,
                            configs):
    import re
    task_root = "LLMGraph/tasks"
    task_root = os.path.join(task_root,task_name)
    for config in configs:
        config_path = os.path.join(task_root,"configs",config)
        if not os.path.exists(config_path):
            print(config_path)
            continue
        data_dst = os.path.join(config_path,"data")
        article_meta_info = readinfo(os.path.join(data_dst,"article_meta_info.pt"))
        regex = r'LLMGraph/tasks/(.*)/data/'
        regex_generated = f'LLMGraph/tasks/{task_name}/configs/{config}/data/generated_article/'
        for article in article_meta_info.values():
            if "generated_article" in article["path"]:
                article["path"] = regex_generated+article["path"].split("/")[-1]
            else:
                task_ori = re.search(regex, article["path"]).group(1)
                article["path"] = article["path"].replace(f"/{task_ori}/",f"/{task_name}/")
            assert os.path.exists(article["path"])
        writeinfo(os.path.join(data_dst,"article_meta_info.pt"),article_meta_info)


    
if __name__ == "__main__":
    
    run_experiments()