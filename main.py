import argparse
import os
os.environ["OPENAI_API_KEY"] = "sk-cIrswbIqa0Josai1263f2aA963A94d7eA224B843BaBb4dE4"
os.environ["OPENAI_API_BASE"] = "https://api.aigcbest.top/v1"
import shutil
from LLMGraph.executor import Executor
parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器


parser.add_argument('--config', 
                    type=str, 
                    default="test_config", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="cora", 
                    help='The task setting for the LLMGraph')  # 添加参数

parser.add_argument("--api_path",
                    type=str,
                    default="LLMGraph/llms/api.json",
                    help="The default path of apis json.")

parser.add_argument("--build",
                    action='store_true',
                    default=False,
                    help="start the building process")




args = parser.parse_args()  # 解析参数



if __name__ == "__main__":
    
    args = {**vars(args)}
    

    if args["build"]:
        
        executor = Executor.from_task(args)
        executor.run()
    
