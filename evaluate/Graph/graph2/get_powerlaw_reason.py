import torch
import json

import powerlaw
from LLMGraph.utils.io import writeinfo, readinfo
from evaluate.article.build_graph import build_citation_graph
import json
import os
import torch
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[1]
    # assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except Exception as e:
        data_dir = os.path.join(os.path.dirname(data_dir),f"{os.path.basename(data_dir).split('.')[0]}.json")
    try:
        with open(data_dir,'r',encoding = 'utf-8') as f:
            data_list = json.load(f)
            return data_list
    except Exception as e:
        print(e)
        raise ValueError("file type not supported", data_dir)

def writeinfo(data_dir,info):
    file_type = os.path.basename(data_dir).split('.')[1]
    if file_type == "pt":
        torch.save(info,data_dir)
    elif file_type == "json":
        with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)
    else:
        raise ValueError("file type not supported")



def count_fitness_centual(DG:nx.DiGraph):
    degree_centrality = nx.degree_centrality(DG)
    # 计算入度
    in_degrees = dict(DG.in_degree())

    # 创建DataFrame以便分析
    data = {
        'node': list(degree_centrality.keys()),
        'degree_centrality': list(degree_centrality.values()),
        'in_degree': list(in_degrees.values())
    }

    df = pd.DataFrame(data)

    # 按入度分组，并计算度中心性的平均值
    grouped_avg = df.groupby('in_degree')['degree_centrality'].mean().reset_index()
    return grouped_avg

def count_author_distribution_time(config,
                                   llm = "gpt4-mini",
                                   task = "llm_agent_4"):
    import re
    from collections import Counter
    

   
    # config = ""

    path = f"LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt"

    meta_info = torch.load(path)
    if len(meta_info)<1000:
        return
    idx = 0
    article_meta_graph = build_citation_graph(meta_info)
    
    article_cite_map = {}
    title_node_map = {
        title:str(idx) for idx,title in enumerate(meta_info.keys())
    }
    dfs = []
    for article, info in list(meta_info.items()):
        if "citations" not in info.keys():
            idx += 1
            continue
        general_cite_subgraph = article_meta_graph.subgraph(list(article_meta_graph.nodes())[:idx])
        cited_items = info["searched_items"]
        regex = r"Title: (.*?)/n"
        cited_items_infos = re.findall(regex, cited_items)
        cite_items_infos_map = {}
        for title in cited_items_infos:
            try:
                cite_items_infos_map[title] = int(general_cite_subgraph.in_degree(
                    title_node_map[title]
                ))
            except:
                pass

        cited_articles = info["citations"]
        general_cites_past = [general_cite_subgraph.in_degree(n) for n in general_cite_subgraph.nodes()]
        searched_cites = list(cite_items_infos_map.values())
        idx += 1
        actual_cites = []

        for cited_article in cited_articles:
            if cited_article in cite_items_infos_map.keys():
                actual_cites.append(cite_items_infos_map[cited_article])
        
        frequency_general = Counter(general_cites_past)
        frequency_search = Counter(searched_cites)
        frequency_cites = Counter(actual_cites)
        fitness = count_fitness_centual(general_cite_subgraph)

        # 仅仅统计search到的pi分布
        cite_pi_map = {}
        general_sum = sum([k*v for k,v in frequency_general.items()])
        search_sum = sum([k*v for k,v in frequency_search.items()])
        fitness_sums = []
        for k,v in frequency_general.items():
            if k in fitness.index:
                fitness_ = fitness.loc[k,"degree_centrality"]
            else:
                fitness_ = 1
            fitness_sums.append(k*v*fitness_)
        fitness_sum = sum(fitness_sums)
        for cite_num in frequency_search.keys():
            try:
                estimate_ba = frequency_general[cite_num]*cite_num/general_sum
                estimate_local = frequency_search[cite_num]*cite_num/search_sum
                if cite_num in fitness.index:
                    estimate_fit = frequency_general[cite_num]*cite_num*fitness.loc[cite_num,"degree_centrality"]/fitness_sum
                else:
                    estimate_fit = estimate_ba
                actual = frequency_cites.get(cite_num,0)/sum(frequency_cites.values())
                cite_pi_map[cite_num] = {
                    "estimate_ba":estimate_ba,
                    "estimate_local":estimate_local,
                    "estimate_fit":estimate_fit,
                    "actual":actual
                }
            except:
                continue
        df = pd.DataFrame().from_dict(cite_pi_map)
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all = df_all.groupby(df_all.index).mean().T
    df_all = df_all.sort_index()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建图形
    plt.figure()
    # 绘制直线
    # plt.plot(list(cite_actual_estimate.keys()), y, label='diff_local', color='blue')
    # plt.plot(list(cite_actual_estimate.keys()), [np.average(v["estimate_ba"])-np.average(v["actual"]) for v in cite_actual_estimate.values()], label='diff_ba', color='green')
    plt.plot(df_all.index, df_all["estimate_ba"].values, label='estimate_ba', color='blue')
    plt.plot(df_all.index, df_all["estimate_local"].values, label='estimate_local', color='green')
    plt.plot(df_all.index, df_all["estimate_fit"].values, label='estimate_fit', color='black')
    plt.plot(df_all.index, df_all["actual"].values, label='actual', color='red')

    # 添加标题和标签
    plt.legend()

    # 添加网格
    plt.grid(True)
    root = f"{root_dir}/citation_preferential/{task}/{llm}"
    os.makedirs(root,exist_ok=True)
    df_all.to_csv(f"{root}/{config}.csv")
    # 添加图例
    plt.savefig(f"{root}/{config}.png")


def cosine_similarity(vec1, vec2):
    # 将向量转换为numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算各自的模
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 计算余弦相似度
    if norm1 == 0 or norm2 == 0:
        return 0.0  # 防止除以0
    else:
        return dot_product / (norm1 * norm2)

def count_estimate_distance(config,llm = "gpt4-mini",task = "llm_agent_4",estimate = "estimate_local"):
    from scipy import stats
    if not os.path.exists(f"{root_dir}/citation_preferential/{task}/{llm}/{config}.csv"):return None
    df = pd.read_csv(f"{root_dir}/citation_preferential/{task}/{llm}/{config}.csv")
    # # 计算KS距离
    ks_statistic, p_value = stats.ks_2samp(df["actual"].values, 
                                           df[estimate].values)
    print(config)
    print(f"KS Statistic: {ks_statistic}, P-value: {p_value}")
    return ks_statistic
    # 计算KS距离
    # estimate = df["estimate_local"].values+ df["estimate_ba"].values
    # cosine =  cosine_similarity(df["actual"].values, 
    #                                       estimate)
    # small = df[df["estimate_local"] < df["actual"]].shape[0]
    # print(config)
    # print(f"rate: {small/df.shape[0]}")

def count_author_distribution():
    import re
    from collections import Counter
    config = "search_shuffle_base_vllm"
    # config = "search_shuffle_base_vllm_recall100f"
    # config = "search_shuffle_base_vllm_recall200f"
    # config = ""

    path = f"LLMGraph/tasks/llm_agent_5/configs/{config}/data/article_meta_info.pt"

    meta_info = torch.load(path)
    

    author_cite_map = {} #author:[cite list],[search list]
    
    idx =0
    article_meta_graph = build_citation_graph(meta_info)
    for article, info in meta_info.items():
        if "citations" not in info.keys():
            idx += 1
            continue
        cited_items = info["searched_items"]
        regex = r"Title: (.*?)/nTopic:.*?/nCited: (/d+)/n"
        cited_items_infos = re.findall(regex, cited_items)
        cite_items_infos_map = {}
        for title,cite in cited_items_infos:
            cite_items_infos_map[title] = int(cite)
        cited_articles = info["citations"]
        author = info["author_ids"][0]
        if author not in author_cite_map:
            author_cite_map[author] = {
                "general":[],
                "search":[],
                "cited":[],
            }
        general_cite_subgraph = article_meta_graph.subgraph(list(article_meta_graph.nodes())[:idx])
        general_cites_past = [general_cite_subgraph.in_degree(n) for n in general_cite_subgraph.nodes()]

        idx += 1
        author_cite_map[author]["general"].extend(general_cites_past)
        author_cite_map[author]["search"].extend(list(cite_items_infos_map.values()))
        for cited_article in cited_articles:
            if cited_article in cite_items_infos_map.keys():
                author_cite_map[author]["cited"].append(cite_items_infos_map[cited_article])
    
    author_p = {}
    for author in author_cite_map.keys():
        cited_infos = author_cite_map[author]["cited"]
        # 统计频率
        frequency = Counter(cited_infos)
        frequency_search = Counter(author_cite_map[author]["search"])
        frequency_general = Counter(author_cite_map[author]["general"])
        search_sum = sum(author_cite_map[author]["search"])
        general_sum = sum(author_cite_map[author]["general"])
        cite_p_e = {}

        for cite_num in frequency_search.keys():
            estimate_local = frequency_search[cite_num]*cite_num/search_sum # 局部演化 estimate
            estimate_ba = frequency_general[cite_num]*cite_num/general_sum
            try:
                cite_p_e[cite_num] = {
                    "actual":frequency.get(cite_num,0)/sum(frequency.values()),
                    "estimate_local":estimate_local,
                    "estimate_ba":estimate_ba
                }
            except:
                cite_p_e[cite_num] = {
                    "actual":0,
                    "estimate_local":estimate_local,
                    "estimate_ba":estimate_ba
                }
        # 输出结果
        author_p[author] = cite_p_e

    cite_actual_estimate ={}
    for author in author_p.keys():
        for cite_num in author_p[author]:
            if cite_num not in cite_actual_estimate.keys():
                cite_actual_estimate[cite_num] =  {
                    "actual":[],
                    "estimate_ba":[],
                    "estimate_local":[]
                }
            cite_actual_estimate[cite_num]["actual"].append(author_p[author][cite_num]["actual"])
            cite_actual_estimate[cite_num]["estimate_ba"].append(author_p[author][cite_num]["estimate_ba"])
            cite_actual_estimate[cite_num]["estimate_local"].append(author_p[author][cite_num]["estimate_local"])
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    cite_actual_estimate = {int(key): value for key, value in cite_actual_estimate.items()}
    cite_actual_estimate = dict(sorted(cite_actual_estimate.items()))
    y = [np.average(v["estimate_local"])-np.average(v["actual"]) for v in cite_actual_estimate.values()]
    # 创建图形
    plt.figure()
    # 绘制直线
    # plt.plot(list(cite_actual_estimate.keys()), y, label='diff_local', color='blue')
    # plt.plot(list(cite_actual_estimate.keys()), [np.average(v["estimate_ba"])-np.average(v["actual"]) for v in cite_actual_estimate.values()], label='diff_ba', color='green')
    plt.plot(list(cite_actual_estimate.keys()), [np.average(v["estimate_local"]) for v in cite_actual_estimate.values()], label='estimate_local', color='blue')
    plt.plot(list(cite_actual_estimate.keys()), [np.average(v["estimate_ba"]) for v in cite_actual_estimate.values()], label='estimate_ba', color='green')
    plt.plot(list(cite_actual_estimate.keys()),[np.average(v["actual"]) for v in cite_actual_estimate.values()],
             label='actual', color='red')

    # 添加标题和标签
    plt.title('Line Plot Example')
    plt.legend()

    # 添加网格
    plt.grid(True)

    # 添加图例
    plt.savefig(f"{root_dir}/citation_preferential/{config}.png")

def plot_cite_actual():
    import matplotlib.pyplot as plt
    import numpy as np
    cite_actual_estimate = readinfo(f"{root_dir}/cite_actual_estimate.json")
    
    cite_actual_estimate = {int(key): value for key, value in cite_actual_estimate.items()}
    cite_actual_estimate = dict(sorted(cite_actual_estimate.items()))
    y = [np.average(v["estimate"])-np.average(v["actual"]) for v in cite_actual_estimate.values()]
    # 创建图形
    plt.figure()
    # 绘制直线
    plt.plot(list(cite_actual_estimate.keys()), y, label='y = 2x + 1', color='blue')

    # 添加标题和标签
    plt.title('Line Plot Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加网格
    plt.grid(True)

    # 添加图例
    plt.savefig(f"{root_dir}/test.png")
    pass


def get_author_qk():
    path = "LLMGraph/tasks/llm_agent/configs/search_shuffle_base/evaluate/author.pt"
    querys = torch.load(path)
    querys = dict(filter(lambda x:x[1]!=[], querys.items()))

    path = "LLMGraph/tasks/llm_agent/configs/search_shuffle_base/data/article_meta_info.pt"
    article_meta_data = torch.load(path)

    import numpy as np
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders.text import TextLoader
    from langchain_core.prompts import PromptTemplate
    from tqdm import tqdm
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = HuggingFaceEmbeddings(model_name="/home/jiarui_ji/.cache/huggingface/transformers/sentence-transformers/all-MiniLM-L6-v2")
    from LLMGraph.loader.article import DirectoryArticleLoader
    text_loader_kwargs={'autodetect_encoding': True}
    article_loader = DirectoryArticleLoader(
                         article_meta_data = article_meta_data,
                         path = "", 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
    docs = article_loader.load()
    prompt_template = PromptTemplate.from_template("""
Title: {title}
Cited: {cited}
Publish Time: {time}
Content: {page_content}""")
    docs_str = [prompt_template.format(**doc.metadata,
                                       page_content = doc.page_content) 
                                       for doc in docs]
    docs_embed = embeddings.embed_documents(docs_str)
    
    relevance_array = []

    for i, author_i in tqdm(enumerate(querys.keys()),
                             "building relevance array..."):
        array_sub = []
        for j, article_j in enumerate(article_meta_data.keys()):
            if i==j:
                array_sub.append(0)
                continue
            embed_i = embeddings.embed_query(",".join(querys[author_i]))
            embed_j = docs_embed[j]
            similarity = cosine_similarity([embed_i], [embed_j])[0][0]
            array_sub.append(similarity)
        relevance_array.append(array_sub)
    relevance_array = np.array(relevance_array)
    relevance_array_all_path = "LLMGraph/tasks/llm_agent/configs/search_shuffle_base/evaluate/relevance_qk.npy"
    np.save(relevance_array_all_path,relevance_array)
    return relevance_array


def plot_loglog_degree(config,color,label,llm = "gpt4-mini",task = "llm_agent_4"):

    from collections import Counter
    import powerlaw

    path = f"LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt"

    meta_info = torch.load(path)
    
    article_meta_graph = build_citation_graph(meta_info)
    if llm == "llama8b":
        threshold = -1
    else:
        threshold = 1000
    if len(article_meta_graph.nodes())<threshold:
        return None
    article_meta_graph = article_meta_graph.subgraph(list(article_meta_graph.nodes())[:threshold])

    meta_info = torch.load(path)
    cites = [article_meta_graph.in_degree(n) for n in article_meta_graph.nodes()]
    cites_1 = list(filter(lambda x:article_meta_graph.in_degree(x[0]),
                     list(article_meta_graph.nodes(data=True))))
    cites_title = [_[1]["title"] for _ in cites_1]
    cites_title_ndv = Counter(cites_title)

    xmin = int(max(cites)*0.05)
    if xmin == 0:
        xmin = 1
 

    result = powerlaw.Fit(cites, xmin=xmin, sigma_threshold =.1)
    return (result.D,result.alpha,result.xmin, len(cites_title_ndv), max(cites))
    

def concat_KS_dfs(root_dir):
    estimate_kinds = ["estimate_local", "estimate_ba", "estimate_fit"]
    dfs = {}
    for estimate_kind in estimate_kinds:
        path = f"{root_dir}/loglog/KS_{estimate_kind}.csv"
        df = pd.read_csv(path,index_col = 0)
        dfs[estimate_kind] = df["base"]

    df_all = pd.DataFrame().from_dict(dfs)
    df_all = df_all.T
    df_all.to_csv(f"{root_dir}/loglog/KS_all.csv")


def get_powerlaw_reason_result(root_dir ="evaluate\Graph\graph2\powerlaw_reason"):
    tasks = ["llm_agent_4","llm_agent_5","llm_agent_6"]
    df_Ds =[]
    df_ndvs =[]
    df_largest_indegrees =[]
    colors = [
            'red', 
            'green', 
            'yellow', 
          ]
    labels = [
        'base',
        'recall100',
        'noinfo',
    ]

    config_templates = [
                "search_shuffle_base_{llm}_powerlaw_base",
                "search_shuffle_base_{llm}_recall100f",
                "search_shuffle_base_{llm}_nocitetime",
                ]
    for task in tasks:
        df_D = pd.DataFrame()
        df_ndv = pd.DataFrame()
        df_largest_indegree = pd.DataFrame()
        llms = ["gpt3.5","gpt4-mini","vllm"]
        for llm in llms:
            for config_template,color,label in zip(config_templates,colors,labels):
                config = config_template.format(llm = llm)
                return_val = plot_loglog_degree(config,color,label,llm = llm,task = task)
                if return_val is not None:
                    D,alpha,xmin, ndv, max_cite = return_val
                else:
                    continue
                df_D.loc[llm, label] = D
                df_ndv.loc[llm, label] = ndv
                df_largest_indegree.loc[llm, label] = max_cite
        df_Ds.append(df_D)
        df_ndvs.append(df_ndv)
        df_largest_indegrees.append(df_largest_indegree)
    
    os.makedirs(root_dir,exist_ok=True)
    df_D = pd.concat(df_Ds)
    df_D.to_csv(f"{root_dir}/loglog/D.csv")
    df_ndv = pd.concat(df_ndvs)
    df_ndv = df_ndv.groupby(df_ndv.index).mean()
    df_ndv.to_csv(f"{root_dir}/loglog/ndv.csv")
    df_largest_indegree = pd.concat(df_largest_indegrees)
    df_largest_indegree.to_csv(f"{root_dir}/loglog/largest_indegree.csv")


get_powerlaw_reason_result()