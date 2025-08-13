import os
import torch
from evaluate.Graph.graph1.get_powerlaw_fit import draw_power_law,get_data,build_citation_graph

def shuffle_seed(
                from_path = "LLMGraph/tasks/cora_1/configs/gt/data/article_meta_info.pt",
                to_path = "LLMGraph/tasks/cora_1/configs/fast_llama3_shuffle/data/article_meta_info.pt"):
   
    import random
    from evaluate.article.build_graph import build_citation_graph
    import powerlaw
    random.seed(42)
    # 将原始的citation网络打乱，生成一个类似随机图的结构，节点的出度数量服从原始出度的均值和标准差
    import numpy as np  
    citations = torch.load(from_path)
    nodes = list(citations.keys())
    # 统计所有节点的原始出度        
    original_graph = build_citation_graph(citations)
    original_out_degrees = [original_graph.out_degree(n) for n in original_graph.nodes()]
    total_out_degree = sum(original_out_degrees)
    num_nodes = len(nodes)
    
    # 计算平均出度，用于均匀分配
    avg_out_degree = total_out_degree / num_nodes if num_nodes > 0 else 0
    
    # 按照原始图的out degree重新分配，均匀分配每个node的degree
    new_cited_articles = {title: [] for title in nodes}
    
    # 为每个节点分配平均出度数量的边
    for title in nodes:
        # 计算该节点应该有的出度数量（取整）
        target_out_degree = int(avg_out_degree)
        # 如果平均出度有小数部分，随机分配额外的边
        if random.random() < (avg_out_degree - target_out_degree):
            target_out_degree += 1
            
        # 随机选择目标节点，避免自环
        possible_targets = [n for n in nodes if n != title]
        if possible_targets and target_out_degree > 0:
            # 随机选择目标节点，确保不超过可能的目标数量
            num_targets = min(target_out_degree, len(possible_targets))
            selected_targets = random.sample(possible_targets, num_targets)
            new_cited_articles[title] = selected_targets

    # 更新每个节点的cited_articles
    for title in nodes:
        citations[title]["cited_articles"] = new_cited_articles[title]
    # # 为每个节点随机分配cited_articles，保证没有自环
    # for idx, title in enumerate(nodes):
    #     possible_targets = [n for n in nodes if n != title]
    #     k = min(new_out_degrees[idx], len(possible_targets))
    #     if k == 0:
    #         citations[title]["cited_articles"] = []
    #         continue
    #     citations[title]["cited_articles"] = random.sample(possible_targets, k)
        
    # 构建citation graph
    graph = build_citation_graph(citations)
    print(graph.number_of_nodes(),graph.number_of_edges())

    # 计算入度分布
    degree_list = [graph.in_degree(n) for n in graph.nodes()]

    # powerlaw拟合
    xmin = 1
    results = powerlaw.Fit(list(degree_list), 
                                discrete=True,
                                sigma_threshold=.1)
    
    print(results.power_law.alpha)
    print(results.power_law.D)
    torch.save(citations,to_path)
    
def plot_powerlaw_fit_shuffle(root_dir ="evaluate/Graph/graph1"):
    config_templates_map ={
    "Cora":[
        ("cora_1","fast_gpt3.5_shuffle"),
        ("cora_1","fast_gpt4-mini_shuffle"),
        ("cora_1","fast_llama3_shuffle"),
        ("cora_1","gt_shuffle"),
    ],
    # "Citeseer":[
    #     ("citeseer_1","fast_gpt3.5_shuffle"),
    #     ("citeseer_1","fast_gpt4-mini_shuffle"),
    #     ("citeseer_1","fast_llama3_shuffle"),
    #     ("citeseer_1","gt_shuffle"),
    # ]
    }
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
        "fast_llama3_shuffle": "GT-shuffle",
        # "gt":"Seed"
    }
    os.makedirs(root_dir,exist_ok=True)
    graphs = []
    for task_name in config_templates_map.keys():
        commands = config_templates_map[task_name]
        if task_name == "LLM-Agent":
            max_nodes = 1000
        else:
            max_nodes = 5000
        for command in commands:
            task_name,config = command
            article_meta_info,author_info = get_data(task_name,config)
            graph = build_citation_graph(article_meta_info)
            if graph.number_of_nodes() > max_nodes:
                graph = graph.subgraph(list(graph.nodes())[:max_nodes])
            print(graph.number_of_nodes(),graph.number_of_edges())
            graphs.append(graph)

    draw_power_law(graphs,"citation_3",config_templates_map = config_templates_map, llm_name_map = llm_name_map,root_dir =root_dir)
   

# for from_path, to_path in zip(["LLMGraph/tasks/cora_1/configs/gt/data/article_meta_info.pt",
#                                "LLMGraph/tasks/citeseer_1/configs/gt/data/article_meta_info.pt"],
#                               ["LLMGraph/tasks/cora_1/configs/fast_llama3_shuffle/data/article_meta_info.pt",
#                                "LLMGraph/tasks/citeseer_1/configs/fast_llama3_shuffle/data/article_meta_info.pt"]):
#     shuffle_seed(from_path, to_path)      
    
plot_powerlaw_fit_shuffle()