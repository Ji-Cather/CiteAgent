# from LLMGraph.manager import ForumManager
# from LLMGraph.involvers import System,Tool,Search_forum_topk
from LLMGraph.message import Message
import asyncio
from LLMGraph.prompt.cora import cora_prompt_registry

from LLMGraph.output_parser import (OutputParseError,
                                    cora_output_parser_registry
                                    )
from langchain.agents.agent import Agent as langchainAgent

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain

from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.agents import AgentExecutor
from typing import Any, List, Optional, Tuple, Union,Dict
from pydantic import root_validator
from langchain.schema import AgentAction,AgentFinish

from langchain.callbacks.manager import (
    Callbacks,
)

from langchain_community.chat_models import ChatOpenAI

from LLMGraph.agent.Langchain_agent_executor import Graph_AgentExecutor

from LLMGraph.memory import SocialMemory,BaseMemory,memory_registry
from LLMGraph.manager import ArticleManager

import re
import random
import copy

from langchain_core.runnables import Runnable
from . import agent_registry
from .tool_openai_agent import create_openai_tools_agent
# from LLMGraph.llms import APIKeyPool
from openai import RateLimitError,AuthenticationError

from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
 


@agent_registry.register("article_agent")
class ArticleAgent(langchainAgent):

    name :str 
    family_num:int=0
    infos : dict 
    social_memory : SocialMemory 
    write_memory: BaseMemory

    max_retry:int = 5 #访问api

    # social_network: dict = {}
    mode : str = "group_discuss" # 控制llm_chain 状态（reset_state中改）
    
    # 这个是为了更改llm_chain
    llm_loader: Any 
    llm: BaseLanguageModel
    
    write_article_agent:Runnable = None
    
    llm_config:dict={
        "self":{},
        "memory":{}
    } # 存储llm的参数设置，为了后续改api用
    
    def __init__(self,  **kwargs):
        infos = kwargs.pop("infos")
        if "extra_info" in infos.keys():
            infos["extra_info"] = "\nYou sincerely believe this information:{}".format(infos.get("extra_info"))
        else:
            infos["extra_info"] = ""
        llm_loader = kwargs.pop("llm_loader")
        # memory = ActionHistoryMemory(llm=kwargs.get("llm",OpenAI()))
        super().__init__(infos = infos,
                         **kwargs)
        self.llm_loader = llm_loader
    
    class Config:
        arbitrary_types_allowed = True
   
    
    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ConvoOutputParser()
    
    @classmethod
    def create_prompt(
        cls, 
        prompt
        ):
        # not used
        # only for abstract method initilization
        return prompt
    
    @property 
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "
    
    # 这个返回值，限制output_parser的返回内容
    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ["return_values"]
    
    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        
        return list(set(self.llm_chain.input_keys)-{"agent_scratchpad"})
    
    
    # @root_validator()
    # def validate_prompt(cls, values) :
    #     """Validate that prompt matches format."""
    #     prompt = values["llm_chain"].prompt
    #     if "agent_scratchpad" not in prompt.input_variables:
    #         prompt.input_variables.append("agent_scratchpad")
    #         if isinstance(prompt, ChoosePromptTemplate):
    #             prompt.template += "\n{agent_scratchpad}"
    #     return values
    
    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs
    
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)

        return self.output_parser.parse(full_output)
    
    
    def chain(self, prompt: PromptTemplate,verbose:bool=False) -> LLMChain:  
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=verbose
        )
        
        
    def reset_llm(self,
                  llm):
        self.llm = llm
        if self.mode == "write_article":
            self.reset_state(self.mode,allowed_tools=self.allowed_tools)
        
    def reset_memory_llm(self,llm):
        self.social_memory.reset_llm(llm)
    
    def reset_state(self,
                    mode = "group_discuss",
                    verbose :bool = False,
                    allowed_tools :Optional[List[str]] = []):
       
        if self.mode == mode : return
        self.mode = mode
        self.allowed_tools = allowed_tools
        
        if self.mode == "write_article":
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.prompts import MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{role_description}"),
                    ("system", "you have discussed your research with other researchers. {discussion_others}"),
                    ("system","{current_paper}"),
                    ("human", "{instruction}"),
                    MessagesPlaceholder(variable_name="searched_info",optional=True),
                    ("ai","{agent_scratchpad}")
                ]
            )
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7,
                                  openai_api_key = self.llm.openai_api_key)
            self.write_article_agent = create_openai_tools_agent(self.llm, allowed_tools, prompt)
        else:
            prompt = cora_prompt_registry.build(mode)
            output_parser = cora_output_parser_registry.build(mode)
            self.llm_chain = self.chain(prompt=prompt,verbose=verbose)
            self.output_parser = output_parser
            
        
        

            
    # 发消息给别人
    def send_message(self,
                    step_type,
                      sendcontent : dict= {},
                      tool_response = [],
                      receivers : list = [], # agent_id: agent_name
                      # 下面三个参数，仅在step_type == "social_network"时用到
                      conver_num = 0, # 表示本轮更新前的conver数
                      continue_dialogue : bool = True ,    # 记录对话是否继续 
                      ):
        
        kargs={ "content":sendcontent,
                "sender":self.name,
                "receivers":receivers,
                "message_type":step_type,
                "tool_response": tool_response,
                "conver_num":conver_num,
                "continue_dialogue":continue_dialogue}
        

        sendmessage = Message(
            **kargs
        ) #给别人发的信息
            
        self.social_memory.add_post_meesage_buffer(messages=[sendmessage])
            
    
    # 发信息给其他agent
    def post_messages(self):
        post_messages = self.social_memory.post_meesages()
        return post_messages
    
    
    def receive_messages(self,messages:List[Message]=[]):
        self.social_memory.add_message(messages=messages)
           
    
    # 更新自己的记忆（发消息给自己）
    def update_social_memory(self,
                      step_type,
                      selfcontent : dict= {},
                      receivers : list = [], # agent_id: agent_name
                      tool_response = [],
                      conver_num = 0, # 表示本轮更新前的conver数    
                      continue_dialogue : bool = True,     # 记录对话是否继续,
                      ):
        
        if selfcontent is None:
            return 
        
        kargs={ "content": selfcontent,
                "sender":self.name,
                "receivers":receivers,
                "message_type":step_type,
                "tool_response": tool_response,
                "conver_num":conver_num,
                "continue_dialogue":continue_dialogue}
        selfmessage = Message(
            **kargs
        ) #给别人发的信息
            
        self.social_memory.add_message(messages=[selfmessage])
        
        
    def step(self, 
             prompt_inputs:dict, 
             tools=[],
             ) -> Message:
        """Generate the next message"""


        
        executor = Graph_AgentExecutor(
            agent = self,
            tools = tools,
            verbose = True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

        response = None
        for i in range(self.max_retry):
            try:
                response = executor(prompt_inputs)
                break
            except OutputParseError as e:
                print(e)
                print("Retrying...")
                continue
            except RateLimitError as e:
                e
        if response is None:
            # raise ValueError(f"{self.name} failed to generate valid response.")
            return {"output":f"{self.name} failed to generate valid response.",
                    "thought":""
                    }

        return response
    
    
    # 异步版本的step
    async def astep(self, 
             prompt_inputs:dict, 
             ) -> Message:
        """Generate the next message"""

        if self.mode == "write_article":
            agent = self.write_article_agent
        else:
            agent = self
        
        executor = Graph_AgentExecutor(
            agent = agent,
            tools = self.allowed_tools,
            verbose = True,
            return_intermediate_steps=True,
        )

        response = None
        for i in range(self.max_retry):
            try:
                response = await executor.ainvoke(prompt_inputs)
                if response is not None:
                    break
            except AuthenticationError as e:
                if isinstance(self.llm,OpenAI) or isinstance(self.llm,ChatOpenAI):
                    api_key = self.llm.openai_api_key
                    self.llm_loader.invalid(api_key)
                    memory_llm, llm_base = self.llm_loader.get_llm(self)
                    self.reset_memory_llm(memory_llm)
                    self.reset_llm(llm_base)
                print(e)
                print("Retrying...")
                continue
            except OutputParseError as e:
                print(e)
                print("Retrying...")
                continue
            except Exception as e:
                print(e)
                continue
                
            
            
        if response is None:
            return {"return_values":{"output":f"{self.name} failed to generate valid response.",
                    "thought":"",
                    "finish":False
            }}
        return response
            

    # 这里定义初始的agent，
    # 如果需要修改prompt,用self.reset_prompt()
    @classmethod
    def from_llm_and_tools(
        cls,
        name,
        infos,
        agent_configs,
        social_network:list,
        llm_loader
    ) -> langchainAgent:
        """Construct an agent from an LLM and tools."""
        
        
        agent_configs = copy.deepcopy(agent_configs)
        agent_llm_config = agent_configs.get('llm')
        
        init_mode = "group_discuss"
        prompt = cora_prompt_registry.build(init_mode)
        output_parser = cora_output_parser_registry.build(init_mode)

        llm = llm_loader.get_llm(llm_configs = agent_llm_config)
        
        
        
        social_memory_config = agent_configs.get('social_memory')
        social_memory_config.update({"social_network":copy.deepcopy(social_network)})
        memory_llm_configs = social_memory_config.get("llm",agent_llm_config)
        memory_llm = llm_loader.get_llm(llm_configs = memory_llm_configs)
        social_memory_config["llm"] = memory_llm
        social_memory_config["memory_llm_configs"] = memory_llm_configs
        social_memory_config["llm_loader"] = llm_loader
        social_memory_config["name"] = infos["name"]
        social_memory = memory_registry.build(**social_memory_config)
        
        
        write_memory_config = agent_configs.get('write_memory')
        # memory_llm_configs = write_memory_config.get("llm",agent_llm_config)
        # memory_llm = llm_loader.get_llm(llm_configs = memory_llm_configs)
        # write_memory_config["llm"] = memory_llm
        # write_memory_config["memory_llm_configs"] = memory_llm_configs
        # write_memory_config["llm_loader"] = llm_loader
        write_memory_config["name"] = infos["name"]
        # write_memory_config_temp = copy.deepcopy(write_memory_config)
        write_memory = memory_registry.build(**write_memory_config)
        
        
        
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
        )
        
        # memory = load_memory(memory_config = memory_config)
        return cls(
            #llm_chain = llm_chain,
            output_parser = output_parser,
            llm = llm,
            llm_loader = llm_loader,
            llm_chain = llm_chain,
            name = name,
            infos = infos,
            social_memory = social_memory,
            write_memory = write_memory,
            llm_config = agent_llm_config,
            mode = init_mode
        )
   
     
    
   
    # 异步的communication过程，包括四种动作（一种放弃）
    async def async_communication(self,
                                  research_content,
                                  article_manager
                                  ):
        target_agent_name = await self.choose_researcher(research_content,
                                                         article_manager)
        return await self.group_discuss(target_agent_name,article_manager)

        
        
    async def choose_researcher(self,
                                research_content,
                                article_manager:ArticleManager
                                ):
        self.reset_state(mode="choose_researcher")
        past_context = self.social_memory.retrieve_recent_chat("all")
        researcher_infos = self.social_memory.get_researchers_infos()
        
        prompt_inputs={
            "role_description":article_manager.get_author_description(self.name),
            "research_topic":research_content["topic"],
            "past_context":past_context,
            "researcher":researcher_infos,
            }
        response = await self.astep(prompt_inputs)
        response = response.get("return_values",{})
        try:
            name = response.get("researcher","")
            for candidate in self.social_memory.social_network.keys():
                if candidate in name:
                    return candidate
        except:
            candidate = random.choice(self.social_memory.social_network.keys())
        return candidate
  
    # 这里加一个recent chat
    async def group_discuss(self,
                      author_name,
                      article_manager:ArticleManager):
        self.reset_state(mode="group_discuss")
        character_1 = article_manager.get_author_description(self.name)
        character_2 = article_manager.get_author_description(author_name)
        past_context = self.social_memory.retrieve_recent_chat("all")
        cur_context = self.social_memory.retrieve_recent_chat(author_name)
        
        prompt_inputs = {
            "character_1":character_1,
            "character_2":character_2,
            "past_context":past_context,
            "cur_context":cur_context,
        }
        response = await self.astep(prompt_inputs)
        response = response.get("return_values",{})
        try:
            context = response.get("communication","")
            self.send_message("social",
                              sendcontent =context,
                              receivers=[author_name],
                              )
            self.update_social_memory(step_type="social",
                                      selfcontent=context,
                                      receivers=[author_name])
        
        except Exception as e:
            print(e)
        
                    
    async def topic_generation(self,
                      article_manager:ArticleManager,
                      research_content
                      ):
        self.reset_state(mode="get_topic")
        past_context = self.social_memory.retrieve_recent_chat("all")
        researcher_infos = self.social_memory.get_researchers_infos()
        
        prompt_inputs={
            "role_description":article_manager.get_author_description(self.name),
            "research_topic":research_content["topic"],
            "past_context":past_context,
            "researcher":researcher_infos,
            }
        
        response = await self.astep(prompt_inputs)
        response = response.get("return_values",{})
        try:
            action = response.get("action","")
            if action == "discuss":
                return False
            else:
                research_content["topic"] = response["topic"]
                research_content["keyword"] = response["keyword"]
                research_content["abstract"] = response["abstract"]
                return research_content["finish"]
        
        except Exception as e:
            print(e)
            return False
   
    async def write_article(self,
                      article_manager:ArticleManager,
                      research_content,
                      tools:list,
                      min_citations = 10,
                      max_refine_round = 5,
                      all_discussion = ""
                      ):
        self.reset_state("write_article",
                         allowed_tools=tools)
        
        discussion_prompt ="""
You have discuss your research with other researchers. Other researchers include:
{researcher}

Your discussion are listed as follows: 
{past_context}
"""

        current_paper_template = """
The version of your paper now: 
    Title: {title}
    Keywords: {keywords}
    Abstract: {abstract}
    Citations: {citations}
    Finish: False
"""
        search_info_template = """
You have already searched the following keywords, so avoid searching them again! 
{searched_info}

Search other keywords instead!!
"""

        output_format ="""
        
- If you want to generate a version of paper, your paper should cite at least {min_citations} papers. You should respond in the following format:
    Title: (The title of your paper, be concise and specific)
    Keywords: (The keywords for your next paper)
    Abstract: (The topic content of the paper you want to write)
    Citations: (List; This should include all the titles of the papers you cite. You should include the papers you have searched.)
     
Respond:
""".format(min_citations = min_citations)
        
        researcher_infos = self.social_memory.get_researchers_infos()
        
        refine_round = 0
        searched_keywords = []
        citation_article_names =[]
        
        try:
            while(len(citation_article_names) < min_citations and \
                  refine_round < max_refine_round):
                
                prompt_inputs ={
            "role_description":article_manager.get_author_description(self.name),
            "discussion_others": discussion_prompt.format(researcher = researcher_infos,
                                                          past_context = all_discussion),
            "current_paper": current_paper_template.format(
                title = research_content.get("title",""),
                keywords = ",".join(research_content["keyword"]),
                abstract = research_content["topic"]+"\n"+research_content["abstract"],
                citations = "\n".join([f"{idx+1}. \"{name}\"" \
                    for idx,name in enumerate(citation_article_names)])
            ),
            "instruction": output_format
        }          
                
                if len(searched_keywords)>0:
                    from langchain_core.messages import (
                        
                        convert_to_messages,
                    )
                    searched_info = search_info_template.format(
                        searched_info = "\n".join([f"{idx+1} . {query}" \
                            for idx, query in enumerate(searched_keywords)])
                    )
                    prompt_inputs["searched_info"] = convert_to_messages([searched_info])
                response = await self.astep(prompt_inputs)
            
                article_info = response.get("return_values",{})
                citations = article_manager.filter_citations(article_info["citation"])
                citation_article_names.extend(citations)
                citation_article_names = list(set(citation_article_names)) # 去重
                article_info["citation"] = citation_article_names
                research_content.update(article_info)
                steps = response.get("intermediate_steps")[-1]
                for step in steps:
                    if isinstance(step,OpenAIToolAgentAction):
                        query = step.tool_input["query"]
                        searched_keywords.append(query)
                searched_keywords = list(set(searched_keywords))

                research_content["success"] = True
                refine_round += 1
            return research_content
        except Exception as e:
            print(e)
            return research_content