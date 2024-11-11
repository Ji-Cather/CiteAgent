import networkx as nx
import os
import gzip
import random
from datetime import datetime
import torch
from evaluate.build_graph import calculate_effective_diameter

def build_citation_network(file_path):
   
    # Initialize a directed graph
    G = nx.DiGraph()

    # 读取 `cit-HepTh.txt.gz` 文件，添加边
    with gzip.open(os.path.join(file_path, 'cit-HepTh.txt.gz'), 'rt') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('#'):
                continue
            
            # 从每行提取源节点和目标节点
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)

    # 读取 `cit-HepTh-dates.txt.gz` 文件，添加时间戳
    with gzip.open(os.path.join(file_path, 'cit-HepTh-dates.txt.gz'), 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            # 从每行提取节点和时间戳
            node, timestamp = line.strip().split()
            node = int(node)
            
            # 将时间戳添加到节点的属性中
            if node in G.nodes:
                G.nodes[node]['timestamp'] = timestamp
    # Filter out nodes without a timestamp
    nodes_with_timestamp = [node for node in G.nodes if 'timestamp' in G.nodes[node]]

    # Create a subgraph containing only nodes with timestamps
    G_filtered = G.subgraph(nodes_with_timestamp).copy()
    return G_filtered


def split_graph_month(G):
    # 打开压缩文件
    monthly_graphs = {}
    for node,node_info in G.nodes(data=True): 
        # 转换timestamp为datetime对象
        date_time = datetime.strptime(node_info['timestamp'], '%Y-%m-%d')
        
        # 以 (year, month) 作为key
        month_key = (date_time.year, date_time.month)
        # 将此paper_id添加至对应月份的图
        monthly_graphs[month_key].add_node(node)

    sorted_data = dict(sorted(monthly_graphs.items(), key=lambda item: item[0]))
    return sorted_data

def calculate_effective_diameter():
    G = build_citation_network("evaluate/kdd2003")
    graph_map = split_graph_month(G)
    graph_now = nx.DiGraph()
    diameter_info = {}

    for time, graph in graph_map.items():
        print(time)
        graph_now.add_nodes_from(graph.nodes())
        graph_now.add_edges_from(graph.nodes())
        diameter = calculate_effective_diameter(graph_now)
        diameter_info[time] = {
            "diameter":diameter,
            "nodes":len(graph_now.nodes),
            "edges":len(graph_now.edges)
        }
    torch.save(diameter_info,
                os.path.join(f"LLMGraph/tasks/kdd2003/evaluate",f"diameter.pt"))
    
calculate_effective_diameter()