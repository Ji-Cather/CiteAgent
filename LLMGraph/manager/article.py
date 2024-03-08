""" load the basic infos of authors"""

import json
import os
from pydantic import BaseModel
from . import manager_registry as ManagerRgistry
from typing import List,Union
from copy import deepcopy
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

import random

@ManagerRgistry.register("article")
class ArticleManager(BaseModel):
    """
        manage infos of different community.
    """
    article_meta_data :dict = {}
    author_data :dict = {}
    path_title_map = {}
    retriever : VectorStoreRetriever
    article_dir: str
    generated_article_dir: str
    meta_article_path:str
    author_path:str
    article_loader: DirectoryLoader
    db: FAISS
    topic_agents:dict = {} # topic: [agent_name]
    
    search_kwargs :dict={"k": 6}
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def load_data(cls,
                  article_meta_path,
                  author_path,
                  article_dir,
                  generated_article_dir,
                  task_path,
                  config_path,
                  search_kwargs= {"k":6}):
        article_meta_path = os.path.join(task_path,article_meta_path)
        author_path = os.path.join(task_path,author_path)
        article_dir = os.path.join(task_path,article_dir)
        assert os.path.exists(article_meta_path),"no such file path: {}".format(article_meta_path)
        with open(article_meta_path,'r',encoding = 'utf-8') as f:
            article_meta_data = json.load(f)
            
        assert os.path.exists(author_path),"no such file path: {}".format(author_path)
        with open(author_path,'r',encoding = 'utf-8') as f:
            author_data = json.load(f)

        assert os.path.exists(article_dir),"no such file path: {}".format(article_dir)
        text_loader_kwargs={'autodetect_encoding': True}
        article_loader = DirectoryLoader(article_dir, 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
        
    
        
        generated_article_dir = os.path.join(os.path.dirname(config_path),
                                             generated_article_dir)
        embeddings = OpenAIEmbeddings()
        path_title_map = {}
        if os.path.exists(generated_article_dir):
            
            generated_article_loader = DirectoryLoader(generated_article_dir, 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
            generated_docs = generated_article_loader.load()
            
        else:
            os.makedirs(generated_article_dir)
            generated_docs =[]
        
        for article_name,article_info in article_meta_data.items():
            base_name = os.path.basename(article_info["path"])
            path_title_map[base_name] = article_name
        
        docs = article_loader.load()
        if len(generated_docs)>0:
            docs =[*docs,*generated_docs]
            
        for doc in docs:
            title_name = path_title_map.get(os.path.basename(doc.metadata['source']))
            assert title_name is not None
            doc.metadata["title"] = title_name
            
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        # if generated_db is not None:
        #     db.merge_from(generated_db)
            
        retriever = db.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

        topic_agents = {}
        for agent_name, agent_info in author_data.items():
            for topic in agent_info["expertises"]:
                if topic not in topic_agents.keys():
                    topic_agents[topic] = [agent_name]
                elif agent_name not in topic_agents[topic]:
                    topic_agents[topic].append(agent_name)
            
        return cls(
           article_meta_data = article_meta_data,
           author_data = author_data,
           path_title_map = path_title_map,
           retriever = retriever,
           db = db,
           article_loader = article_loader,
           article_dir = article_dir,
           generated_article_dir = generated_article_dir,
           search_kwargs = search_kwargs,
           author_path = author_path,
           meta_article_path = article_meta_path,
           topic_agents = topic_agents
           )
        
    def update_db(self):
        assert os.path.exists(self.generated_article_dir)
        text_loader_kwargs={'autodetect_encoding': True}
        generated_article_loader = DirectoryLoader(self.generated_article_dir, 
                         glob="*.txt", 
                         loader_cls=TextLoader,
                         show_progress=True,
                         loader_kwargs=text_loader_kwargs)
        generated_docs = generated_article_loader.load()
        
        docs = self.article_loader.load()
        if len(generated_docs)>0:
            docs =[*docs,*generated_docs]
            
        for doc in docs:
            title_name = self.path_title_map.get(os.path.basename(doc.metadata['source']))
            assert title_name is not None
            doc.metadata["title"] = title_name
        
        embeddings = OpenAIEmbeddings()
        
        self.db = FAISS.from_documents([*docs,*generated_docs], embeddings)
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs=self.search_kwargs)
        

    
    def write_generated_articles(self,articles):
        
        root = self.generated_article_dir
        max_idx = 0
        num_article = 0
        for config_path in os.listdir(root):
            regex = r"paper_cora_(\d+).txt"
            import re
            try:
                idx = int(re.search(regex,config_path).group(1))
                max_idx = idx if max_idx<idx else max_idx
            except:
                continue
            
        for config_path in os.listdir(self.article_dir):
            regex = r"paper_cora_(\d+).txt"
            import re
            try:
                idx = int(re.search(regex,config_path).group(1))
                max_idx = idx if max_idx<idx else max_idx
            except:
                continue
            
        for idx,article in enumerate(articles):
            if not article["success"]:
                continue
            title = article["title"]
            abstract = article["abstract"]
            del article["abstract"]
            path_new_article = os.path.join(root,f"paper_cora_{max_idx+idx+1}.txt")
            article.update({
                "path": path_new_article,
                "cite":0
            })
            try:
                """update publication"""
                assert title not in self.article_meta_data.keys(), f"generated replicate article!! \n {title}"
                self.article_meta_data[title] = article
                assert not os.path.exists(path_new_article),f"{path_new_article} exists!!"
                with open(path_new_article,"w") as f:
                    f.write(title + " "+ abstract)
                self.path_title_map[os.path.basename(path_new_article)] = \
                    title
                
                co_authors_article = article["author"]
                
                for author in article["author"]:
                    assert author in self.author_data.keys(),f"unknown author {author}"
                    self.author_data[author]["articles"].append(title)
                    self.author_data[author]["publications"] +=1
                    co_authors = self.author_data[author].get("co_authors",[])
                    for co_author_a in co_authors_article:
                        if co_author_a != author and \
                            co_author_a not in co_authors:
                                co_authors.append(author)
                    self.author_data[author]["co_authors"] = co_authors
                
                    
                """update citations"""
                for cite_article in article["citation"]:
                    assert cite_article in self.article_meta_data.keys(), f"unknown article {cite_article}"
                    self.article_meta_data[cite_article]["cite"] +=1
                    for author in self.article_meta_data[cite_article]["author"]:
                        self.author_data[author]["citations"] +=1
                        
                with open(self.meta_article_path,'w',encoding = 'utf-8') as f:
                    json.dump(self.article_meta_data, f, indent=4,separators=(',', ':'),ensure_ascii=False)
                    
                with open(self.author_path,'w',encoding = 'utf-8') as f:
                    json.dump(self.author_data, f, indent=4,separators=(',', ':'),ensure_ascii=False)
                
                num_article +=1
                
            except Exception as e:
                print(e)
                continue
        
        return num_article
        
    def write_and_update_db(self,articles):
        num = self.write_generated_articles(articles=articles)
        self.update_db()
        return num
        
     
    def get_author_description(self,
                             author_name):
        try:
            infos = self.author_data[author_name]

            template="""\
You are a researcher. Your name is {name}. Your research interest is {expertises}.\
"""         
            expertises = ",".join(list(infos["expertises"].keys()))
            role_description = template.format(
                expertises = expertises,
                name = infos["name"]
            )
            return role_description
        except Exception as e:
            print(e)
            return ""
        
    def filter_citations(self,
                         citations_str:str) -> List[str]:
        citations_articles = citations_str.split("\n")
        
        citation_names = []
        for citation_article in citations_articles:
            if citation_article.strip() == "":
                continue
            docs = self.retriever.invoke(citation_article)
            for doc in docs:
                if doc.metadata["title"].lower() in citation_article.lower():
                    citation_names.append(doc.metadata["title"])
                    break
                
        return citation_names
    
    def get_topic_agents(self,topic):
        
        return self.topic_agents.get(topic,[])
    
    def get_most_cooperated_author(self,
                                   authors_topic:List[str],
                                   author_num:int =5):
        first_author = random.choice(authors_topic)
        authors = [first_author]

        ## bfs 
        queue_authors = [first_author]
        while len(authors) < author_num and len(queue_authors) > 0:
            author_name = queue_authors.pop()
            for co_author in self.author_data[author_name].get("co_authors",[]):
                if co_author not in authors:
                    authors.append(co_author)
                    queue_authors.append(co_author)
            
        return authors