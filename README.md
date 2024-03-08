# LLMGraph

Effective economic policy is crucial for avoiding adverse economic phenomena such as inflation, resource monopolization and etc. We propose an economic simulation framework based on LLM-Agents. Specifically, LLM-based agents can engage in interactions, exploration, and decision-making within the LLMGraph simulation framework. 

To refine economic policy parameters, we propose the Policy Optimization Finding algorithm (POA) with custom optimization objectives. The realism and effectiveness of simulation by LLMGraph is validated through Turing tests.


## LLMGraph Framework

Before we begin, please set your openai_api_keys in "LLMGraph\llms\api.json", and format it like:
```json
[
    "sk-***",
    "sk-***"
]
```

Then create the experiment, and install the required packages:
    ```
    pip install -i "requirements.txt"
    ```


- To start simulation in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```cmd
    python main.py --task public_housing --config "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#housetype" --simulate
    ```


- You can run this script to reproduce the results in paper
    ```cmd
    python start.py
    ```
    - Remember to comment out the data and task_name except for the corresponding experiment.

- If you want optimize with certern kind of policy parameters, run it simply by (the max_samples should not exceed the number of runned experiments, the minimum number required for optimizing is 30)
    ```cmd
    python main.py --task public_housing -optimize --optimize_regressor_threshold 0.3 --optimize_regressor_max_samples 60
    ```