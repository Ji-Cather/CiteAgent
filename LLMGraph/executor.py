import asyncio
import logging
from typing import List


from .initialization import (load_environment,
                             prepare_task_config)
from LLMGraph.llms import APIKeyPool
import platform
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


# 删掉load agent，因为environment中不止agent参与，不限制参与类型


class Executor():
    def __init__(self,
                 environment,
                 ex_idx:str):
        self.environment = environment
        self.ex_idx = ex_idx# 标识实验的index

    @classmethod
    def from_task(cls, 
                  args:dict):
        """Build an LLMGraph from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task
        task_config,config_path,task_path = prepare_task_config(args["config"],args["task"])
        
        if platform.system()=='Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        import time
        import os
        save_dir = task_config.pop("save_root_dir","")
        
        time_stamp = time.time()
        save_dir = os.path.join(task_path,
                                f"{save_dir}/{time_stamp}")
        
        
        
        
        llm_loader = APIKeyPool(args["api_path"])
        
        env_config = task_config.pop('environment')
        env_config["llm_loader"] = llm_loader
        
        env_config["task_path"] = task_path
        env_config["config_path"] = config_path
        
        environment = load_environment({**env_config})
        
        
        return cls(environment = environment,
                   ex_idx = time_stamp)


    
    def run(self):
        """Run the environment from scratch until it is done."""
                    
        
        while not self.environment.is_done():
            self.environment.step()
            
        return self.ex_idx

            

    def reset(self):
        self.environment.reset()
      