from pydantic import BaseModel, Field
from typing import List, Tuple, Set

from langchain.schema import AgentAction,BaseMessage
from datetime import datetime
from pydantic import BaseModel
from typing import Union

# 这里导致prompt template返回类型有问题
class Message(BaseModel):
    timestamp: float
    message_type: str
    content: Union[dict,str]
    output_keys: List[str] = []
    importance_rate: float = 0
    relation_rate :float = 0
    sender: str
    receivers: List[str] # receiver names
    tool_response: List[Tuple[AgentAction, str]] = Field(default=[])
    conver_num:int = 0 #记录对话次数
    continue_dialogue : bool = True # 记录对话是否继续
    
    
    def update_attr(self,**kwargs):
        for key,value in kwargs.items():
            self.__setattr__(key,value)
    
    def __init__(self,**kwargs):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        output_keys = kwargs.pop("output_keys",False)
        if not output_keys:
            content = kwargs.get("content")
            if isinstance(content,dict):
                output_keys = list(content.keys())
            else:
                output_keys = []
                
        super().__init__(timestamp = timestamp,
                         output_keys = output_keys,
                         **kwargs)
        
    def sort_rate(self):
        return self.timestamp+self.importance_rate+self.relation_rate

    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"
    
    def __str__(self):
        if isinstance(self.content,dict):
            content_str = [f"{key.capitalize()}:{self.content[key]}"for key in self.output_keys]
            return "\n".join(content_str)
        elif isinstance(self.content,str):
            return self.content
        else:
            raise ValueError()
        
    


# class TenantMassage(Message):
#     make_choice:bool = False
#     chosed_housing_idx:int = -1