
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 

import numpy as np

import os
def plot_degree_compare(graph_maps:dict,
                save_root:str):
    
    fig, ax = plt.subplots(1, len(graph_maps), figsize=(14, 6))
    data_combined = []
    for graph in graph_maps.values():
        data_combined += [d for n, d in graph.degree()]
    
    bins = np.linspace(min(data_combined), max(data_combined), 30)

    for idx, graph_name in enumerate(graph_maps.keys()):
        graph = graph_maps[graph_name]
        assert isinstance(graph, nx.DiGraph)
        # 获取节点的度
        degrees_G = list(dict(graph.in_degree(weight='count')).values())
        # 绘制度的概率密度分布函数
        ax[idx].hist(degrees_G, bins=bins, density=True, alpha=0.6, color='b')
        ax[idx].set_title(f'{graph_name} Graph Degree PDF')
        ax[idx].set_xlabel('In Degree')
        ax[idx].set_ylabel('Density')


