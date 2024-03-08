import os
from typing import Dict

import yaml

from LLMGraph.prompt.prompt import prompt_registry
from LLMGraph.manager import manager_registry
from LLMGraph.environments import env_registry
from LLMGraph.memory import memory_registry

def load_memory(memory_config: Dict,
                llm,
                llm_loader):
    memory_config["llm"] =llm
    memory_config["llm_loader"] =llm_loader
    return memory_registry.build(memory_config)


def load_environment(env_config: Dict) :
    env_type = env_config.pop('env_type', 'rent')
    return env_registry.build(env_type, **env_config)



def load_manager(manager_config: Dict,manager_type) :
    return manager_registry.load_data(manager_type, ** manager_config)
    

    
def prepare_task_config(task,
                        data):
    """Read the yaml config of the given task in `tasks` directory."""
    all_data_dir = os.path.join(os.path.dirname(__file__), 'tasks')
    task_path = os.path.join(all_data_dir,data)
    if not os.path.exists(task_path):
        all_datas = []
        for data_name in os.listdir(all_data_dir):
            if os.path.isdir(os.path.join(all_data_dir, task)) \
                and data_name != "__pycache__":
                all_datas.append(data_name)
        raise ValueError(f"Data {data} not found. Available datas: {all_datas}")
    
    all_task_dir = os.path.join(task_path, "configs")
    config_path = os.path.join(all_task_dir, task)
    config_path = os.path.join(config_path, 'config.yaml')
    
    if not os.path.exists(config_path):
        all_tasks = []
        for task in os.listdir(all_task_dir):
            if os.path.isdir(os.path.join(all_task_dir, task)) \
                and task != "__pycache__":
                all_tasks.append(task)
        raise ValueError(f"Task {task} not found. Available tasks: {all_tasks}")
    
    if not os.path.exists(config_path):
        raise ValueError("You should include the config.yaml file in the task directory")
    task_config = yaml.safe_load(open(config_path))

    return task_config,config_path,task_path





def load_prompt(prompt_type: str) :
    config={}
    return prompt_registry.build(prompt_type, **config)


