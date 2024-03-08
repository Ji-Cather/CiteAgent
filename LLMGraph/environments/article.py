import asyncio
import logging
from typing import List, Deque, Optional

from LLMGraph.agent import ArticleAgent
import json
from LLMGraph.manager import ArticleManager
from LLMGraph.involvers.tool import create_retriever_article_tool
from LLMGraph.llms import APIKeyPool

from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random


from langchain.tools import Tool

@EnvironmentRegistry.register("article")
class ArticleEnvironment(BaseEnvironment):
    """
    A environment implementing the logic of conversation.

    Args:
        agents: tenant_manager
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """
    article_manager: ArticleManager

    tools : List[Tool] = []
    
    agent_groups:list = [] # 存储生成各个article的agents
    
    
    agent_configs = {} # 存储agent的config
    
    # 对于api调换的Loader类
    llm_loader: APIKeyPool
    
    article_write_configs :dict = {
        "min_citations":5,
        "paper_num_per_round":10,
        "author_num":5,
        "paper_num":200
        }
    
    article_written_num:int = 0
    

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rule,**kwargs):
        
        article_manager_configs = kwargs.pop("managers").pop("article")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        article_manager = ArticleManager.load_data(**article_manager_configs,
                                         task_path = task_path,
                                         config_path=config_path)
        
        article_write_configs = kwargs.pop("article_write_configs")
        
        retriever = article_manager.retriever
        retriever_tool = create_retriever_article_tool(
            retriever,
            "search",
    "Search for relevant papers, so as to refine your paper. \
These papers should be included in your paper's citations if you use them in your paper. \
Your paper should cite at least {min_citations} papers!".format_map(article_write_configs),
        )
        
        super().__init__(rule=rule, 
                         article_manager = article_manager,
                         agent_configs = agent_configs,
                         tools = [retriever_tool],
                         article_write_configs = article_write_configs,
                         **kwargs)

    
    
        
    def reset(self) -> None:
        """Reset the environment"""

        self.rule.reset(self)
       
        self.cnt_turn = 0

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        return self.article_written_num >= self.article_write_configs["paper_num"] \
            or self.cnt_turn >= self.max_turns
        
        
    async def generate_topic_one_group(self,
                                      group_id,
                                      communication_num = 10):
                                      
            """
            the async run parse of tenant(tenant_id) communication.
            return: the receivers, self(if continue_communication)
            """
            agents = self.agent_groups[group_id]["authors"]
            research_content = self.agent_groups[group_id]["content"]
            idx = 0
            while idx < communication_num:
                agent = agents[idx%len(agents)]
                if ((idx+1)%len(agents)==0):
                    finish_generation = await agent.topic_generation(self.article_manager,
                                                                       research_content)
                    if finish_generation: break
                await agent.async_communication(research_content,self.article_manager)
                    
                self.update_social_net(agent)
                idx += 1
            
    async def write_article_one_group(self,
                                      group_id):
                                      
            """
            the async run parse of tenant(tenant_id) communication.
            return: the receivers, self(if continue_communication)
            """
            agent_first_author = self.agent_groups[group_id]["authors"][0]
            assert isinstance(agent_first_author,ArticleAgent)
            research_content = self.agent_groups[group_id]["content"]
            all_discussion = self.collect_all_discussions(self.agent_groups[group_id]["authors"])
            research_content = await agent_first_author.write_article(self.article_manager,
                                             research_content=research_content,
                                             tools=self.tools,
                                             all_discussion=all_discussion,
                                             min_citations=self.article_write_configs["min_citations"])
            self.agent_groups[group_id]["content"] = research_content
            
            
    #test 测试用 要改
    def communication(self):
        
        async def run_parallel():
            if len(self.agent_groups) >0:
                await asyncio.gather(*[self.generate_topic_one_group(idx,
        communication_num = self.article_write_configs["communication_num"]) 
                                       for idx in range(len(self.agent_groups))])
            
        asyncio.run(run_parallel()) # 需要进行讨论的tenant
            

    def write(self):
        
        async def run_parallel():
            if len(self.agent_groups) >0:
                await asyncio.gather(*[self.write_article_one_group(idx) for idx in range(len(self.agent_groups))])
            
        asyncio.run(run_parallel()) # 需要进行讨论的tenant
    
    
    def group_assign_topic(self,
                   article_number = 10,
                   author_number = 5):
        """_summary_

        Args:
            article_number (int, optional): the number of articles to be generated. Defaults to 10.
            author_number (int, optional): the number of authors for topic discussion (per article). Defaults to 5.
        """
        topics = list(self.article_manager.topic_agents.keys())
        group_agents = []
        
        for i in range(article_number):
            #random sample
            topic = random.choice(topics)
            authors_topic = self.article_manager.get_topic_agents(topic)
            
            if author_number > len(self.article_manager.author_data.keys()):
                authors = list(self.article_manager.author_data.keys())
            else:
                authors = self.article_manager.get_most_cooperated_author(
                    authors_topic=authors_topic,
                    author_num=author_number
                )
            if len(authors) ==1: # 暂时不支持单作者
                other_authors = copy.deepcopy(list(self.article_manager.author_data.keys()))
                other_authors.remove(authors[0])
                if len(other_authors)>1:
                    authors.append(random.choice(other_authors))
                else:authors = list(self.article_manager.author_data.keys())
                
            author_agents = []
            for author in authors:
                sn = {}
                for s_name in authors:
                    if author != s_name:
                        sn[s_name] = self.article_manager.author_data[s_name]
                        
                author_agent = ArticleAgent.from_llm_and_tools(author,
                                                             self.article_manager.author_data[author],
                                                             self.agent_configs,
                                                             sn,
                                                             self.llm_loader)
                author_agents.append(author_agent)
                
            article ={
                "authors":author_agents,
                "content":{
                    "topic": topic,
                    "keyword":[],
                    "abstract":"",
                    "citation":[],
                    "title":"",
                    "author":[agent.name for agent in author_agents],
                    "success":False
                    }
            }
            group_agents.append(article)
            
        self.agent_groups = group_agents
        
    
    
    def test_group(self,
                   article_number = 10,
                   author_number = 5):
        """_summary_

        Args:
            article_number (int, optional): the number of articles to be generated. Defaults to 10.
            author_number (int, optional): the number of authors for topic discussion (per article). Defaults to 5.
        """
        group_agents = []
        for i in range(article_number):
            #random sample
            if author_number < len(self.article_manager.author_data.keys()):
                authors = random.sample(self.article_manager.author_data.keys(),
                                    author_number)
            else:
                authors = list(self.article_manager.author_data.keys())
            author_agents = []
            for author in authors:
                sn = {}
                for s_name in authors:
                    if author != s_name:
                        sn[s_name] = self.article_manager.author_data[s_name]
                        
                author_agent = ArticleAgent.from_llm_and_tools(author,
                                                             self.article_manager.author_data[author],
                                                             self.agent_configs,
                                                             sn,
                                                             self.llm_loader)
                author_agents.append(author_agent)
                
            article ={
                "authors":author_agents,
                "content":{
                    "topic": "data mining",
                    "keyword":[],
                    "abstract":"",
                    "citation":[],
                    "title":"",
                    "author":[agent.name for agent in author_agents],
                    "success":False
                    }
            }
            group_agents.append(article)
        self.agent_groups = group_agents
        
    def collect_all_discussions(self,agents:List[ArticleAgent]):
        discussion = []
        for agent in agents:
            all_discussion = agent.social_memory.retrieve_recent_chat("all")
            discussion.append(all_discussion)
        return "\n".join(discussion)
            
    def step(self):
        self.cnt_turn += 1 
        
        self.group_assign_topic(
            article_number = self.article_write_configs["paper_num_per_round"],
            author_number = self.article_write_configs["author_num"])
        
        self.communication()
        # 接下来是写论文的过程
        self.write()
        
        self.update_article_manager()
        
        self.agent_groups = []
        
    def update_article_manager(self):
        
        num = self.article_manager.write_and_update_db([agent_group["content"] for agent_group in self.agent_groups])
        self.article_written_num += num
            
    def update_social_net(self,agent):
        assert isinstance(agent, ArticleAgent)
        return self.rule.post_messages(environment = self)

        

    