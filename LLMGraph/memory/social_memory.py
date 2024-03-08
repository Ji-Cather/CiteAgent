from typing import List,Union,Dict

from pydantic import Field

from LLMGraph.message import Message

from . import memory_registry,summary_prompt_default
from .base import BaseMemory

from langchain.chains.llm import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel, root_validator
import re
from typing import Any



from .summary import SummarizerMixin


@memory_registry.register("social_memory")
class SocialMemory(BaseMemory,SummarizerMixin):

    
    reflection:bool = False # 若设置为true,则触发分类reflection  
    
    social_network: dict = {} # id:{name:,view:,dialogues：List[Message],chat_history, dialogues_pt}
    
    # 设置各类消息的buffer大小数
    summary_threshold:int = 5 # 每次总结后，再多5条就触发一次总结 -> 记忆库内
    
    # 想要发出的message信息，在发出后清空，加入messages中
    post_message_buffer: List[Message] = []
    
    mail:List[Message] = [] # 用于socialnetwork的接收信息

    

    
    def __init__(self,**kwargs):
        social_network = kwargs.pop("social_network")
        for id,info in social_network.items():
            info["dialogues_pt"] = -1       
            info["chat_history"] = ""
            info["comment"] = ""
            info["dialogues"] = []

        
        super().__init__(social_network = social_network,
                         **kwargs)
    
    def add_message(self, 
                    messages: List[Message]):
        
        for message in messages:
            if message.sender == self.name:
                receive_ids = message.receivers
            else:
                receive_ids = [message.sender]
            for receive_id in receive_ids:
                if receive_id not in self.social_network.keys():
                    self.social_network[receive_id] = {"name":list(message.sender.values())[0],
                                                        "relation":"stranger",
                                                        "dialogues":[],
                                                        "dialogues_pt":-1,
                                                        "chat_history":"",
                                                        "comment":""}
                    
                if "dialogues" in self.social_network[receive_id].keys():
                    self.social_network[receive_id]["dialogues"].append(message)
                else:
                    self.social_network[receive_id]["dialogues"] = [message]
                    self.social_network[receive_id]["dialogues_pt"] = -1
                
    def post_meesages(self):

        post_messages = self.post_message_buffer.copy()
        self.post_message_buffer = []
        return post_messages
        
    def add_post_meesage_buffer(self, 
                    messages: List[Message]):
        self.post_message_buffer.extend(messages)

        
    def topk_message_default(self,
                             messages:List[Message],
                             k=5)->List[Message]:
        messages.sort(key=lambda x: x.sort_rate(),reverse=True)
        return messages[:k] if k<len(messages) else messages
    
 

    def to_string(self, 
                  messages:List[Message],
                  add_sender_prefix: bool = False,
                  ) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    f"{list(message.sender.values())[0]}:{str(message)}"
                    if list(message.sender.values())[0] != ""
                    else str(message)
                    for message in messages
                ]
            )
        else:
            return "\n".join([str(message) for message in messages])
  

    def reset(self) -> None:
        for id,info in self.social_network.items():
            info["dialogues_pt"] = -1       
            info["chat_history"] = ""
            info["comment"] = ""
            info["dialogues"] = []


    ###############         一系列的 retrive memory rule       ##############
    
    #  调用各类 retrive 方法
    async def retrieve_memory(self,
                              agent_ids:list)->str:
        memory = {}
        for agent_id in agent_ids:
            memory_agent = self.retrieve_recent_chat(agent_ids=[agent_id])
            memory[agent_id] = memory_agent
        return memory
        
    def retrieve_recent_chat(self,
                             agent_ids: Union[List,str] = "all"):
        if agent_ids == "all":
            agent_ids = list(self.social_network.keys())
       
        if not isinstance(agent_ids,list):
            agent_ids = [agent_ids]
        
        recent_chats = []
        for agent_id in agent_ids:
            if ("dialogues" in self.social_network[agent_id].keys()):
                dialogues_sn = self.social_network[agent_id].get("dialogues")
                recent_chats.append(self.to_string(dialogues_sn))
                    
        return "\n".join(recent_chats)
    
    def get_researchers_infos(self):
        researchers = []
        for id, researcher in self.social_network.items():
            expertises = researcher["expertises"].keys()
            name = researcher["name"]
            expertises = ",".join(expertises)
            researcher_info = f"{id}. {name}: {expertises}"
            researchers.append(researcher_info)
            
        return "\n".join(researchers)
    

            
    
    
    
    
   
     