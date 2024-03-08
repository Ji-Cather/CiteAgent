from typing import List,Union,Dict

from pydantic import Field

from LLMGraph.message import Message

from . import memory_registry,summary_prompt_default
from .base import BaseMemory

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel, root_validator
import re
import datetime

class Atomic_Info(BaseModel):
    content:str=""
    timestamp:datetime.datetime
    importance_score:float =0
    last_accessed_time:datetime.datetime
    @classmethod
    def message_to_atomic_info(cls,message,importance_score):
        if isinstance(message,Message):
            return cls(
                content=message.content,
                timestamp=datetime.datetime.now(),
                importance_score=importance_score,
                last_accessed_time=datetime.datetime.now()
            )
        elif isinstance(message,str):
            return cls(
                content=message,
                timestamp=datetime.datetime.now(),
                importance_score=importance_score,
                last_accessed_time=datetime.datetime.now()
            )
    
    
    
    

@memory_registry.register("fact_communication")
class FactCommunicationMemory(BaseMemory):
    
    fact:Dict[str,str]={}
    communication:List[Atomic_Info]=[]
    llm: BaseLanguageModel
    mail:List[Message]=[]
    post_message_buffer:List[Message] = []
    reflection:bool = False # 若设置为true,则触发分类reflection  
    
    
    def add_message(self, messages: List[Message]) -> None: 
        for message in messages:
            importance_score=self.message_importance(message=message)
            doc=Atomic_Info.message_to_atomic_info(message=message,importance_score=importance_score)
            self.communication.append(doc)

    def add_fact(self,content_type,content):
        self.fact[content_type]=content
        
        
    def message_importance(self,message):
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g.,Praise for the good community environment and excellent location) and 10 is"
            + " extremely poignant (e.g. Various negative news about the community"
            + "), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        if isinstance(message,Message):
            score = self.chain(prompt).run(memory_content=message.content).strip()
        elif isinstance(message,str):
            score = self.chain(prompt).run(memory_content=message).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return float(match.group(1)) / 10
        else:
            return 0.0
        
    def add_post_meesage_buffer(self, 
                    messages: List[Message]):
        self.post_message_buffer.extend(messages)
        
    def add_forum_info(self,forum_infos: List[str]):
        for forum_info in forum_infos:
            importance_score=self.message_importance(message=forum_info)
            doc=Atomic_Info.message_to_atomic_info(message=forum_info,importance_score=importance_score)
            self.communication.append(doc)
            
         
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=False)
    
    
    def run_reflect(self):
        pass
    
    def generate_points(self, n=3): 
        
        points=[[atinfo.last_accessed_time,atinfo]
                for atinfo in self.communication]
        points = sorted(points, key=lambda x: x[0])
        points = [i for last_accessed_time, i in points]

        statements = "\n".join(points)
        
        prompt = PromptTemplate.from_template(
            "{statements}\
            Given only the information above, what are {number} most salient high-level questions we can answer about the subjects grounded in the statements?"
        )
        point = self.chain(prompt).run(statements=statements,number=n).strip()
        return point
    
    def reset(self) -> None:
        self.fact = {}
        self.communication = []
        self.post_message_buffer = []
        self.mail=[]