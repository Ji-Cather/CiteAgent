import json
import os
import torch

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)


def generate_data():
    cora_title = readinfo("data/cora/results.json")
    cora_graph = torch.load("data/cora/citeseer_fixed_sbert.pt")
    
    authors = {}
    articles = {}
    for idx, article in cora_title.items():
        author_a = article["Authors"]
        for author_name in author_a:
            if author_name in authors.keys():
                co_workers = authors[author_name]["co_author_ids"]
                articles = []
                research_region = []
            else:
                co_workers = []
            # for co_a in author_a:
            #     if co_a not
                authors[author_name]["co_author_ids"]
                
def generate_articles():
    cora_titles = readinfo("data/cora/titles.json")
    authors = readinfo("data/cora/authors.json")
    
    articles = {}
    for author_name, author_info in authors.items():

        for article_name in author_info["articles"]:
            if article_name not in articles.keys():
                for title in cora_titles:
                    if article_name in title:
                        absract = title.replace(article_name,"")
                        break
                articles[article_name] = absract
    writeinfo("LLMGraph/tasks/cora/data/article.json",articles)
    
def generate_origin_data():
    cora_titles = readinfo("data/cora/titles.json")
    crawled_authors = readinfo("LLMGraph/tasks/cora/data/author.json")
    crawled_article_infos = readinfo("data/cora/results.json")
    
    articles = {}
    for idx, article_one_zip in enumerate(zip(cora_titles,crawled_article_infos)):
        abstract, article_info = article_one_zip
        
        
    writeinfo("LLMGraph/tasks/cora/origin_data/article.json",articles)
    

    
if __name__ =="__main__":
    generate_articles()