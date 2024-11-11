import networkx as nx
import numpy as np

from networkx.algorithms.similarity import graph_edit_distance
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm 

import os
import torch
import pandas as pd
import random
from evaluate.article.libs.mrqap import MRQAP
from evaluate.article.libs.qap import QAP
from evaluate.article.libs.ols import OLS
from evaluate.article.gcc_size import calculate_gcc_size
import scipy.stats as stats
from typing import Dict
import time
from datetime import datetime,date
from evaluate.visualize.article import (
                                        plot_self_citation,
                                        plot_betas,
                                        plt_topic_given
                                        )
from evaluate.article.calculate_reason import calculate_reason
from evaluate.article.build_graph import (build_author_citation_graph,
                                          update_citation_graph,
                                          build_country_citation_graph,
                                          build_relevance_array,
                                          build_group_relevance_array,
                                        #   build_group_number_array,
                                          assign_topic,
                                          build_citation_group_array,
                                          build_citation_group_array_from_citation,
                                            build_citation_graph,
                                            build_bibliographic_coupling_network,
                                            build_co_citation_graph,
                                            build_co_authorship_network,
                                            )

from evaluate.matrix import calculate_directed_graph_matrix
from evaluate.matrix.base_info import calculate_effective_diameter
from LLMGraph.utils.io import readinfo, writeinfo
from tqdm import tqdm



parser = argparse.ArgumentParser(description='graph_llm_builder')  # 创建解析器
parser.add_argument('--config', 
                    type=str, 
                    default="test_config", 
                    help='The config llm graph builder.')  # 添加参数

parser.add_argument('--configs', 
                    type=str, 
                    default="test_config,test_config_2", 
                    help='a list of configs for the config llm graph builder.')  # 添加参数

parser.add_argument('--task', 
                    type=str, 
                    default="cora", 
                    help='The task setting for the LLMGraph')  # 添加参数

parser.add_argument('--val_dataset', 
                    type=str, 
                    default="cora", 
                    help='The dataset name for the validation')  # 添加参数


parser.add_argument('--xmin', 
                    type=int, 
                    default=3, 
                    help='power law fit xmin')  # 添加参数

parser.add_argument('--threshold', 
                    type=int, 
                    default=10000, 
                    help='the max number of generated papers')  # 添加参数




def print_graph_info(DG:nx.Graph):
    print('Number of nodes', len(DG.nodes))
    print('Number of edges', len(DG.edges))
    print('Average degree', sum(dict(DG.degree).values()) / len(DG.nodes))


def load_graph(model_dir, 
               val_dataset,
               nodes_len:int = None):
    generated_model_path = os.path.join(model_dir,"graph.graphml")
    # 示例：比较两个随机图
    assert os.path.exists(generated_model_path),"The generated graph path doesn't exist"
    # G_generated = nx.read_adjlist(generated_model_path,create_using=nx.DiGraph())
    G_generated = nx.read_graphml(generated_model_path)
    if nodes_len is not None:
        sub_nodes = [str(i) for i in range(nodes_len)]
        G_generated = G_generated.subgraph(sub_nodes).copy()
    print("Generated Graph:", 
          G_generated,
          'Average degree', 
          "{degree:.3f}".format(degree =
            sum(dict(G_generated.degree).values()) / len(G_generated.nodes)))
    G_true = load_test_dataset(val_dataset)
    print(f"True Graph {val_dataset}:", 
          G_true,
          'Average degree', 
          "{degree:.3f}".format(degree =
            sum(dict(G_true.degree).values()) / len(G_true.nodes)))
    return G_true, G_generated

def build_graphs(article_meta_data:dict,
                 author_info:dict,
                 article_num = None,
                 graph_types:list = [ 
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ]):
    
    graphs = {}
    if article_num is not None:
        article_meta_data = dict(list(article_meta_data.items())[:article_num])

    article_graph = build_citation_graph(article_meta_data)
    graphs["article_citation"] = article_graph
   
    # 节点是论文，边是论文间的引用
    
    if "bibliographic_coupling" in graph_types:
        bibliographic_coupling_network = build_bibliographic_coupling_network(
            article_graph
        )
        graphs["bibliographic_coupling"] = bibliographic_coupling_network
    
    if "co_citation" in graph_types:
        co_citation_graph = build_co_citation_graph(article_graph)
        graphs["co_citation"] = co_citation_graph
    
    if "author_citation" in graph_types:
        # 节点是作者， 边是他们的引用
        author_citation_graph = build_author_citation_graph(article_meta_data,
                                                        author_info)
        graphs["author_citation"] = author_citation_graph

    if "country_citation" in graph_types:
        article_graph = update_citation_graph(article_graph,article_meta_data,author_info)
        # article_graph = article_graph.subgraph(list(article_graph.nodes())[:500])
        country_citation_graph = build_country_citation_graph(article_meta_data,
                                                             author_info,
                                                             article_graph)
        # 节点是国家， 边是他们的引用
        graphs["country_citation"] = country_citation_graph

    if "co_authorship" in graph_types:
        co_authorship_network =  build_co_authorship_network(article_meta_data,
                                                             author_info)
        graphs["co_authorship"] = co_authorship_network
    
    for graph_type, graph in graphs.items():
        if isinstance(graph,nx.DiGraph):
            print(f"{graph_type:>20} Graph:", 
              graph,
              'Average degree', 
              "{degree:.3f}".format(degree =
                sum(dict(graph.degree).values()) / len(graph.nodes)),
                'indegree',
                 "{degree:.3f}".format(degree =
                sum(dict(graph.in_degree).values()) / len(graph.nodes)),
                'outdegree',
                 "{degree:.3f}".format(degree =
                sum(dict(graph.out_degree).values()) / len(graph.nodes)))
        else:
            try:
                print(f"{graph_type:>20} Graph:", 
                graph,
                'Average degree', 
                "{degree:.3f}".format(degree =
                    sum(dict(graph.degree).values()) / len(graph.nodes)))
            except:pass
    

    return graphs


def load_test_dataset(val_dataset):
    
    if val_dataset == "cora":
        DG = nx.DiGraph()
        pass
        return DG
    
    if val_dataset == "citeseer":
        DG = nx.DiGraph()
        path = "LLMGraph/tasks/citeseer/data/article_meta_info.pt"
        articles = readinfo(path)
        article_idx_title_map = {}
        for idx,title in enumerate(articles.keys()):
            article_idx_title_map[title] = idx
            DG.add_node(idx,title=title,time=articles[title]["time"])
            
        for title, article_info in articles.items():
            edges =[]
            cited_articles = article_info.get("cited_articles",[])
            title_idx = article_idx_title_map.get(title)
            for cite_title in cited_articles:
                cited_idx = article_idx_title_map.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))  
            DG.add_edges_from(edges)        
        return DG
    
    if val_dataset == "llm_agent":
        DG = nx.DiGraph()
        path = "LLMGraph/tasks/llm_agent/data/article_meta_info.pt"
        articles = readinfo(path)
        article_idx_title_map = {}
        for idx,title in enumerate(articles.keys()):
            article_idx_title_map[title] = idx
            DG.add_node(idx,title=title,time=articles[title]["time"])
            
        for title, article_info in articles.items():
            edges =[]
            cited_articles = article_info.get("cited_articles",[])
            title_idx = article_idx_title_map.get(title)
            for cite_title in cited_articles:
                cited_idx = article_idx_title_map.get(cite_title)
                if cited_idx is not None:
                    edges.append((cited_idx,title_idx))  
            DG.add_edges_from(edges)        
        DG = DG.subgraph(nodes=list(DG.nodes())[:100])
        return DG
    

    

def visualize_article(
                    
                    article_meta_data:dict,
                    author_info:dict,
                    save_dir:str,
                    task:str = "llm_agent",
                    threshold:int =1000):
    if len(article_meta_data) < threshold:
        return 
    article_meta_data = dict(list(article_meta_data.items())[:threshold])
    graphs = build_graphs(article_meta_data,author_info,
                          graph_types=[#"author_citation",
                                    #    "article_citation",
                            # "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                                       ])
    article_graph = graphs["article_citation"]
    
    # topic distribution
    plt_topic_given(task_name=task,article_meta_data=article_meta_data,save_dir=save_dir)
    # 可视化自引用
    plot_self_citation(article_graph,save_dir=save_dir)

    # 可视化原因
    reason_path = os.path.join(save_root,"reason","reason_info.json")
    calculate_reason(article_meta_data,reason_path)

    # distortion analysis
    for method in ["ols","pearson"]:
        country_types = [
                        # "country_all",
                        "country_core",
                        "country_used"
                        ]
        for country_type in country_types:
            distortion_count(article_graph, 
                            article_meta_data,
                            author_info, 
                            article_meta_info_path,
                            save_dir=save_dir,
                            type=country_type,
                            group=False,
                            method = method)  
            distortion_count(article_graph, 
                            article_meta_data,
                            author_info, 
                            article_meta_info_path,
                            save_dir=save_dir,
                            type=country_type,
                            group=False,
                            experiment_ba=True,
                            method = method)  
            distortion_count(article_graph, 
                            article_meta_data,
                            author_info, 
                            article_meta_info_path,
                            save_dir=save_dir,
                            type=country_type,
                            group=False,
                            experiment_er=True,
                            method = method)  
     

    


def build_er_article_meta_graph(article_graph:nx.DiGraph):
    import random
    countrys = readinfo("evaluate/article/country.json")
    countrys_list = []
    for v in countrys.values():
        countrys_list.extend(v)
    countrys_list = countrys_list
    for node,node_info in article_graph.nodes(data=True): # 所有国家的article number same chance
        country = random.choice(countrys_list)
        article_graph.nodes[node]["country"] = [country]
    return article_graph





# 3. 选择key值
def weighted_random_choice(countrys_articles):
    # 1. 计算总和
    total = sum(countrys_articles.values())

    # 2. 归一化概率
    probabilities = {country: count / total for country, count in countrys_articles.items()}
    # 使用random.choices()可以根据权重选择国家
    countries = list(probabilities.keys())
    weights = list(probabilities.values())
    selected_country = random.choices(countries, weights=weights, k=1)[0]
    return selected_country

def build_ba_article_meta_graph(article_meta_data:dict,
                                G:nx.DiGraph,
                                type:str):
    # use all_countrys
    countrys = readinfo("evaluate/article/country.json")
    if type =="country_all":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
    elif type =="country_core":
        countrys_list = countrys["core"]
    elif type =="country_used":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
        countrys_list = countrys_list
    
    countrys_list = [country.lower() for country in countrys_list]
    citation_array = np.zeros((len(countrys_list),len(countrys_list)))

    countrys_articles = {}
    start_point = 90
    points_all = len(article_meta_data)
    start_G = G.subgraph(nodes=list(G.nodes())[:start_point])
    # 用子图 H 初始化一个新的有向图
    new_DG = nx.DiGraph()

    # 将子图中的节点和边添加到有向图中
    new_DG.add_nodes_from(start_G.nodes(data=True))  # 保留节点属性
    new_DG.add_edges_from(start_G.edges(data=True))  # 保留边属性
    start_G = new_DG
    country_node_map = {}
    for node, node_info in start_G.nodes().items():
        for country in node_info["country"]:
            country_node_map[country] = node

    for node in list(article_meta_data.keys())[:start_point]:
        countrys_articles[country] = article_meta_data[node]["cited"]

    titles_add = list(article_meta_data.keys())
    idx = start_point
    edges = 10
    for node in list(G.nodes())[start_point:]:
        countrys = G.nodes[node]["country"]
        for country in countrys:
            if country not in countrys_articles:
                countrys_articles[country] = 0
            countrys_articles[country]+=1

        # country_max = max(countrys_articles, key=countrys_articles.get)
        # 添加点的country ~ country number prob
        # 添加点的edge prob ~ node的 country number prob
        country_max = weighted_random_choice(countrys_articles) # 按照country权重选择新点的country
        start_G.add_node(idx, country = country_max, title = titles_add[idx])
        idx_node = idx
        # 1. 计算总和
        total = sum(countrys_articles.values())
        # 2. 归一化概率
        probabilities = {country: count / total for country, count in countrys_articles.items()}
        for country, prob in probabilities.items():
            edge_num = prob * edges
            if country not in country_node_map.keys():
                start_G.add_node(idx, country = [country], title = titles_add[idx])
                idx +=1
                if idx >= len(article_meta_data):
                    break
                else:
                    continue
            for _ in range(int(edge_num)):
                start_G.add_edge(country_node_map[country], idx_node)
                
        if idx >= len(article_meta_data):
            break
        start_G.add_node(idx, country = [country], title = titles_add[idx])
        idx +=1
        if idx >= len(article_meta_data):
            break
    return start_G





def get_countrys_list(article_meta_data,
                      article_graph,
                      map_index:dict,
                      type:str
                      ):
    countrys = readinfo("evaluate/article/country.json")
    if type =="country_all":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
    elif type =="country_core":
        countrys_list = countrys["core"]
    elif type =="country_used":
        countrys_list = []
        for v in countrys.values():
            countrys_list.extend(v)
        countrys_list = countrys_list
    
    countrys_list = [country.lower() for country in countrys_list]

    group_articles = {}
    ## 计算分组的文章相似度

    for idx, article_info in enumerate(article_meta_data.values()):
        node = map_index[article_info["title"]]
        try:
            countrys = article_graph.nodes[node]["country"]
        except:
            continue
        for country in countrys:
            if country not in countrys_list:
                continue
            if country not in group_articles.keys():
                group_articles[country] = []
            group_articles[country].append(node)
    for country in group_articles.keys():
        group_articles[country] = list(set(group_articles[country]))
    
    # 使用存在article的country
    # countrys_list_all = list(group_articles.keys())
    # countrys_list = list(filter(lambda x: x in countrys_list,countrys_list_all))

    group_number_array = np.zeros((len(countrys_list),len(countrys_list)))
    for country in group_articles.keys():
        country_index = countrys_list.index(country.lower())
        group_number_array[country_index][country_index] = len(group_articles[country])

    return group_number_array,countrys_list

def distortion_count(article_graph:nx.DiGraph,
                     article_meta_data:dict,
                    author_info:dict,
                    article_meta_info_path:str,
                    save_dir:str = "evaluate/article/distortion",
                    group:bool = False,
                    type:str = "country_all",
                    experiment_base: bool = False,
                    experiment_ba:bool = False,
                    experiment_er: bool = False,
                    use_equal_similarity: bool = False,
                    method = "ols"
                    ):
    save_dir_ori = save_dir
    if use_equal_similarity:
        save_dir = os.path.join(save_dir,"equal_similarity")
    
    save_dir = os.path.join(save_dir,f"{method}")
    if experiment_base:
        save_dir = os.path.join(save_dir,"distortion_base")
    elif experiment_er:
        save_dir = os.path.join(save_dir,"distortion_er")
    elif experiment_ba:
        save_dir = os.path.join(save_dir,"distortion_ba")
    else:
        save_dir = os.path.join(save_dir,"distortion_llm")
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    beta_save_root = os.path.join(save_dir,f"beta_dict_{type}.json")
   
    article_graph = update_citation_graph(article_graph,article_meta_data,author_info)

    if use_equal_similarity:
        relevance_array = np.ones((len(article_meta_data),len(article_meta_data)))
    else:
        relevance_array_all_path = os.path.join(save_dir_ori,"relevance_array_all.npy")
        if os.path.exists(relevance_array_all_path):
            relevance_array = np.load(relevance_array_all_path)
        else:
            relevance_array = build_relevance_array(article_meta_data)
            np.save(relevance_array_all_path,relevance_array)

   
    article_meta_data_grouped = {}
    group_key = "topic"
    for title,article in article_meta_data.items():
        if article[group_key] not in article_meta_data_grouped:
            article_meta_data_grouped[article[group_key]] = {}
        article_meta_data_grouped[article[group_key]].update({title:article})
    
    beta_dict = {}
    map_index = {
        title:str(idx) for idx,title in enumerate(article_meta_data.keys())} # keep index map
    
    if beta_dict == {}:
        if "citeseer" in save_dir:
            start_time = datetime.strptime("2004-01", "%Y-%m").date()
            end_time = datetime.strptime("2011-01", "%Y-%m").date()
        else:
            start_time = datetime.strptime("2021-04", "%Y-%m").date()
            end_time = datetime.strptime("2024-08", "%Y-%m").date()
        # start_time = datetime.strptime("2004-01", "%Y-%m").date()
        # end_time = datetime.strptime("2011-01", "%Y-%m").date()
        article_meta_data = dict(filter(
            lambda x: datetime.strptime(x[1]["time"],"%Y-%m").date() <= end_time, 
                article_meta_data.items()
        ))
        index_start = len(article_meta_data)
        for idx,title in enumerate(article_meta_data.keys()):
            if datetime.strptime(article_meta_data[title]["time"],"%Y-%m").date() >= start_time:
                if (index_start > idx):
                    index_start = idx
            else:
                pass
        ## run mrqap for all articles
        time_thunk_size = int((len(article_meta_data) - index_start) // 60)
        beta_list = []
        time_lines = []
        if experiment_ba:
            article_graph = build_ba_article_meta_graph(article_meta_data,article_graph,type)
            
            for idx, node_info in article_graph.nodes(data=True):
                title = node_info["title"]
                article_meta_data[title]["country"] = node_info["country"]
            writeinfo(os.path.join(save_dir, "ba_article_meta_graph.pt"),article_meta_data)

        if experiment_er:
            article_graph = build_er_article_meta_graph(article_graph)
        else:
            pass
        

        for i in tqdm(range(index_start,len(article_meta_data),time_thunk_size),
                    desc=f"run qap for all articles: {type}"): 
            article_meta_data_time =  dict(list(article_meta_data.items())[:i+time_thunk_size])

            group_number_array,countrys_list = get_countrys_list(article_meta_data_time,article_graph,map_index,type)
            article_graph_sub = article_graph.subgraph(list(article_graph.nodes)[:i+time_thunk_size])
            
            if experiment_ba or experiment_er:
                topic_citation_array = build_citation_group_array_from_citation(article_graph_sub,
                                                         countrys_list,
                                                         type)
            else:
                topic_citation_array = build_citation_group_array(
                                                                article_meta_data_time,
                                                                author_info,
                                                                countrys_list,
                                                            type)
                
            topic_relevance_array = build_group_relevance_array(relevance_array,
                                            # article_meta_data_time,
                                            article_graph_sub,
                                            author_info,
                                            article_graph,
                                            countrys_list,
                                            type,
                                            map_index)
            
            betas = run_qap(topic_relevance_array,topic_citation_array,type = method)

            if betas is None and not np.isnan(betas[0]):
                continue
            beta_list.append(betas)
            time_lines.append(list(article_meta_data_time.values())[-1]["time"])
        beta_dict["all"] = {"y":beta_list,
                            "x":time_lines}
        
        writeinfo(beta_save_root,beta_dict)





def run_qap(relevance_array: np.ndarray,
              X,
              type = 'pearson'):
    
    np.random.seed(1)
    NPERMUTATIONS = 100
    #######################################################################
    # QAP
    #######################################################################
    start_time = time.time()
    qap = QAP(Y=relevance_array, X=X, npermutations=NPERMUTATIONS,  diagonal=False,
              type=type)
    qap.qap()
    
    return np.average(qap.betas), np.array(qap.betas).std()



def get_data(task,config):
    data_path = "LLMGraph/tasks/{task}/configs/{config}/data"
    article_meta_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                              "article_meta_info.pt"))
    author_info = readinfo(os.path.join(data_path.format(task=task,config=config),
                                       "author.pt"))
    return article_meta_info,author_info


def plt_distortion(beta_save_root:str,
                   article_meta_data:dict,
                   types:list,
                   save_dir:str ="evaluate/article/distortion"):
    betas_dict = {}
    for type in types:
        beta_save_path = os.path.join(beta_save_root,f"beta_dict_{type}.json")
        assert os.path.exists(beta_save_path)
        betas = readinfo(beta_save_path)
        betas_dict[type] = betas

    key = "CITATION"
    for topic in betas.keys():
        beta_data = []
        error_data = []
        try:
            for idx in range(len(betas[topic]["y"])):
                beta_data.append({
                        type_beta: betas_dict[type_beta][topic]["y"][idx][key][0] 
                        for type_beta in betas_dict.keys()}
                                )
                error_data.append({
                        type_beta: betas_dict[type_beta][topic]["y"][idx][key][1] 
                        for type_beta in betas_dict.keys()}
                                )
        except:
            continue
        time_data = betas_dict[types[0]][topic]["x"]

        plot_betas(time_data,beta_data,save_dir=save_dir,
                   types=types,group_name=topic)
        

def calculate_article_matrix(G:nx.DiGraph,
                            graph_name:str,
                            save_dir:str):
    if graph_name == "article_citation":
        calculate_matrix = [
                           "community",
                        "base_info"            
                        ]
    elif graph_name in ["author_citation",
                        "co_authorship"]:
        calculate_matrix = [
                        "community",
                        "base_info"            
                        ]
    else:
        calculate_matrix = [
                          "community",
                            "base_info"            
                            ]
  
    matrix = calculate_directed_graph_matrix( 
                                        G,
                                        graph_name=graph_name,
                                        type = "article",
                                        calculate_matrix = calculate_matrix)
    
    save_path = os.path.join(save_dir,f"{graph_name}_matrix.csv")
    os.makedirs(save_dir, exist_ok=True)
    matrix.to_csv(save_path)

def calculate_article_power_law(G:nx.DiGraph,
                            graph_name:str,
                            save_dir:str,
                             plt_flag=True,
                             xmin:int = 3):
    from evaluate.matrix import calculate_power_law
    power_law_dfs = calculate_power_law( 
                            G,
                            save_dir=save_dir,
                            graph_name=graph_name,
                            plt_flag=plt_flag,
                            xmin=xmin)
    for degree_type, df in power_law_dfs.items():
        save_path = os.path.join(save_dir,
                                 f"{graph_name}_{degree_type}_power_law.csv")
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_path)



import copy
from collections import defaultdict
from tqdm import tqdm
def calculate_preference(citation_graph,
                         graph_type,
                         save_dir,
                         author_info,
                         article_info):
    if "article_citation" != graph_type:return
    # if os.path.exists(os.path.join(save_dir, "preferences.json")):
    #     return
    experiment_ba = False
    citation_graph = update_citation_graph(citation_graph,article_info,author_info)
    if experiment_ba:
        citation_graph = build_ba_article_meta_graph(article_info,citation_graph,"country_used")


    countrys_info = readinfo("evaluate/article/country.json")
    countrys = [countrys_info[country] for country in countrys_info.keys()]
    c =[]
    for c_ in countrys:
        for _ in c_:
            c.append(_.lower())
    countrys = c
    country_articles = {c:0 for c in countrys} # country: article_num
    coauthorship_articles = {n:copy.deepcopy(country_articles) for n in author_info.keys()} # author:{country:article_num}
    citation_articles = {n:copy.deepcopy(country_articles) for n in author_info.keys()} # author:{country:article_num}
    
    start_node = 200
    if len(citation_graph.nodes)>2000:
        end_node = len(citation_graph.nodes)
    else:
        end_node = 500
    calculate_time_evolution = True # 把按照时间的变化画出来

    if calculate_time_evolution:
        preferences_time = {
            "CPS":[],
            "RPS":[],
            # "RPS_other":[],
            # "RPS_own":[]
        } # rps: [(rps(c_core),rps_pheir,rps_score)]
    preferences = {n:{
                        "CPS":[],
                        "RPS":[],
                        # "RPS_other":[],
                        # "RPS_own":[]
                    } 
                   for n in author_info.keys()}
   
    for node,info in list(citation_graph.nodes.data())[:start_node]:
        for country in info["country"]:
            country_articles[country] += 1

    idx = 0
    core_country_num = len(countrys_info["core"])
    for node, info in tqdm(list(citation_graph.nodes.data())[start_node:end_node],
    "preference"
    ):
        title = info["title"]
        author_ids = article_info[title]["author_ids"]
        
        citation_countrys = {c:0 for c in countrys}
        out_neighbors = citation_graph.successors(node)
        # 遍历每个出边终点节点
        for neighbor in out_neighbors:
            # 获取终点节点的 country 属性
            countries = citation_graph.nodes[neighbor].get('country', [])
            # 统计每个 country 出现的次数
            for country in countries[:1]:
                citation_countrys[country] += 1
        citation_countrys = dict(citation_countrys)

        for author_id in author_ids:
            coauthor_ids = copy.deepcopy(author_ids)
            coauthor_ids.remove(author_id)
            for co_id in coauthor_ids:
                try:
                    coauthorship_articles[author_id][author_info[co_id]["country"].lower()] += 1
                except:continue
            for country in citation_countrys.keys():
                citation_articles[author_id][country] += citation_countrys[country]
            
            

            CPS ={}
            RPS = {}

            for c in countrys:
                CPS[c] = 1
                RPS[c] = 1
              
                country_share = country_articles[c] / sum(country_articles.values())
                
                if coauthorship_articles[author_id][c]==0 or country_share ==0:
                    pass
                else:
                    cps = coauthorship_articles[author_id][c]/ sum(coauthorship_articles[author_id].values())
                    CPS[c] = cps/country_share
                if citation_articles[author_id][c] ==0 or country_share ==0:
                    pass
                else:
                    rps = citation_articles[author_id][c]/ sum(citation_articles[author_id].values())
                    RPS[c] = rps/country_share
                
               
            
                
            preferences[author_id]["CPS"].append(CPS)
            preferences[author_id]["RPS"].append(RPS)
    
        if calculate_time_evolution and idx%5==0:
            
            preferences_processed = {}
            for key in ["CPS","RPS"]:
                ps_mean = []
                for person in preferences.keys():
                    ps_persons = preferences[person][key]
                    ps_persons = pd.DataFrame(ps_persons)
                    
                    
                    # ps_persons = ps_persons.apply(lambda x: x[x != 0].mean() if (x != 0).any() else 0)
                    ps_persons = ps_persons.mean()

                    # ps_persons= (ps_persons > 1).sum()
                    # the preference of one person remains constant
                    # ps_mean[person] = ps_persons / ps_persons.sum()
                    ps_mean.append(ps_persons)
                    
                ps_mean = pd.concat(ps_mean,axis=1).T
                ps_mean = ps_mean[countrys]
                preferences_processed[key] = {
                    "mean":ps_mean.mean().to_dict(),
                    "std":ps_mean.std().to_dict()
                }
                
                values = list(preferences_processed[key]["mean"].values())
                core_rps = values[:core_country_num]
                ph_rps = values[core_country_num:]
                # core_rps = list(filter(lambda x: x!=-1, core_rps))
                # ph_rps = list(filter(lambda x: x!=-1, ph_rps))
                # if len(ph_rps)==0:
                #     continue
                all_rps = ph_rps+core_rps
                preferences_time[key].append((np.average(core_rps),
                                              np.average(ph_rps),
                                              np.mean(all_rps),
                                              np.average(core_rps)/np.average(ph_rps)
                                                )
                                              )
        idx +=1

    preferences_processed = {}
    for key in ["CPS","RPS"]:
        ps_mean = []
        for person in preferences.keys():
            ps_persons = preferences[person][key]
            ps_persons = pd.DataFrame(ps_persons)
            
            
            # ps_persons = ps_persons.apply(lambda x: x[x != 0].mean() if (x != 0).any() else 0)
            ps_persons = ps_persons.mean()

            # ps_persons= (ps_persons > 1).sum()
            if ps_persons.max()==0:continue
            # the preference of one person remains constant
            # ps_mean[person] = ps_persons / ps_persons.sum()
            ps_mean.append(ps_persons)
            
        ps_mean = pd.concat(ps_mean,axis=1).T
        values = ps_mean.mean().to_dict()
        # for k, v in values.items():
        #     if v == 0:
        #         values[k] = 1
        preferences_processed[key] = {
            "mean":values
        }
    if experiment_ba:
        save_dir = os.path.join(save_dir,"preferences_ba")
    os.makedirs(save_dir,exist_ok=True)
    writeinfo(os.path.join(save_dir, "preferences_time.json"),preferences_time)
    writeinfo(os.path.join(save_dir, "preferences.json"),preferences_processed)

def calculate_all_graph_matrix(
                
                article_meta_data:dict,
                author_info:dict,
                save_dir:str,
                
                article_num = None,
                graph_types:list = [
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                xmin:int = 3,
                threshold = 1000
                ):
    if len(article_meta_data) < threshold:
        return 
    article_meta_data = dict(list(article_meta_data.items())[:threshold])
    graphs = build_graphs(article_meta_data,
                 author_info, 
                 article_num = article_num,
                 graph_types = graph_types)
    
    for graph_type,graph in tqdm(graphs.items(),
                                 "calculate all graph matrix"):
        print("calculating", graph_type)
        if graph_type in graph_types:
            
            calculate_diameter_trend(graph,graph_type,save_dir)
            calculate_article_power_law(graph,graph_type,save_dir,plt_flag=True,xmin=xmin)
            calculate_gcc_size(graph,graph_type,save_dir)
            calculate_article_matrix(graph,graph_type,save_dir)
            calculate_preference(graph,graph_type,save_dir,author_info,article_meta_data)
            save_degree_list(graph,save_dir,graph_type)
            


def calculate_densification_power_law_trend(graph,graph_type,save_dir):
    chop_len = 50
    group_size = int(len(graph.nodes)/chop_len)
    d_power_law = {}
    diameter_info = {
        "group_size":group_size,
        "group_len":chop_len,
        "d_power_law":{}
    }
    for i in tqdm(range(0,len(graph.nodes),group_size),"calculate densification power law"):
        sub_graph  = graph.subgraph(list(graph.nodes)[:i+group_size])
        d_power_law[sub_graph.number_of_nodes()] = sub_graph.number_of_edges()
    diameter_info["d_power_law"] = d_power_law
    torch.save(diameter_info,
               os.path.join(save_dir,f"{graph_type}_d_power_law.pt"))
    

def calculate_diameter_trend(graph,graph_type,save_dir):
    chop_len = 50
    group_size = int(len(graph.nodes)/chop_len)
    post_04_nodes = dict(filter(lambda x: datetime.strptime(x[1]["time"], "%Y-%m") >= datetime(2004, 1, 1), graph.nodes(data=True)))
    post_04_graph = graph.subgraph(list(post_04_nodes.keys()))
    diameter_info = {}
    for i in tqdm(range(0,len(graph.nodes),group_size),"calculate diameter"):
        sub_graph  = graph.subgraph(list(graph.nodes)[:i+group_size])
        post_04_k_sub = list(filter(lambda x:x in post_04_nodes.keys(),
            sub_graph.nodes()
        ))
        
        
        try:
            diameter = calculate_effective_diameter(sub_graph)
            node_to_check = list(sub_graph.nodes)[-1]
            time = sub_graph.nodes[node_to_check]["time"]
            sub_graph = nx.Graph(sub_graph)
           
            # 找到所有连通分量并选择最大的
            largest_component = max(nx.connected_components(sub_graph), key=len)
            
            # # 找到最大的完全子图
            # largest_clique = max(nx.find_cliques(sub_graph), key=len)
            
            # sub_coauthor = build_co_authorship_subgraph(sub_graph)
            # # 找到最大的完全子图
            # largest_clique_cc = max(nx.find_cliques(sub_coauthor), key=len)
            # # 找到所有连通分量并选择最大的
            # largest_component_cc = max(nx.connected_components(sub_coauthor), key=len)
            # # 创建子图（包含最大连通分量的所有节点和边）
            try:
                post_04_subgraph = post_04_graph.subgraph(post_04_k_sub)
                post_04_subgraph = nx.Graph(post_04_subgraph)
                graph_04_len = len(post_04_subgraph)
                diameter_04 = calculate_effective_diameter(post_04_subgraph)
                largest_cc_04 = len(max(nx.connected_components(post_04_subgraph), key=len))
                nodes_04 = len(post_04_subgraph.nodes)
                edges_04 = len(post_04_subgraph.edges)
            except Exception as e:
                diameter_04 = 0
                largest_cc_04 = 0
                graph_04_len= 0
                nodes_04 = 0
                edges_04 = 0
            diameter_info[time] = {
                "diameter":diameter,
                "diameter_04":diameter_04,
                # "diameter_lcc":calculate_effective_diameter(sub_graph.subgraph(largest_component)),
                # "diameter_lcc_cc":calculate_effective_diameter(sub_coauthor.subgraph(largest_component_cc)),
                # "diameter_cc":calculate_effective_diameter(sub_coauthor),
                "nodes":len(sub_graph.nodes),
                "edges":len(sub_graph.edges),
                "nodes_04":nodes_04,
                "edges_04":edges_04,
                # "nodes_cc":len(sub_coauthor.nodes),
                # "edges_cc":len(sub_coauthor.edges),
                # "clique_len":len(largest_clique),
                "gcc_len": len(largest_component),
                "gcc_len_04":largest_cc_04,
                "graph_len_04":graph_04_len,
                "graph_len": len(sub_graph),
                # "clique_len_cc":len(largest_clique_cc),
                # "gcc_len_cc": len(largest_component_cc),
                # "graph_len_cc": len(sub_coauthor)
            }
        except Exception as e:
            continue

    torch.save(diameter_info,
               os.path.join(save_dir,f"{graph_type}_diameter.pt"))


def save_degree_list(G:nx.DiGraph,
                    save_dir:str,
                    graph_name:str):
    save_degree_root = os.path.join(save_dir,"degree")
    os.makedirs(save_degree_root,exist_ok=True)
    degree_list = [G.in_degree(n) for n in G.nodes()]
    writeinfo(os.path.join(save_degree_root,f"{graph_name}.json"),degree_list)
    


if __name__ == "__main__":
    args = parser.parse_args()  # 解析参数
    save_root = "LLMGraph/tasks/{task}/configs/{config}/evaluate".format(
        task = args.task,
        config = args.config
    )

    generated_article_dir = "LLMGraph/tasks/{task}/configs/{config}/data/generated_article".format(
        task = args.task,
        config = args.config
    )

    article_meta_info_path = "LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt".format(
        task = args.task,
        config = args.config
    )
    
    article_meta_info,author_info = get_data(args.task,args.config)

    calculate_all_graph_matrix(
                               article_meta_info,
                               author_info,
                               save_root,
                               graph_types=[
                            "article_citation",
                            "bibliographic_coupling",
                            "co_citation",
                            "author_citation", 
                            "country_citation",
                            "co_authorship"
                            ],
                            xmin=args.xmin,
                            threshold=args.threshold)
    visualize_article(article_meta_info,author_info,save_root,task=args.task,threshold=args.threshold)