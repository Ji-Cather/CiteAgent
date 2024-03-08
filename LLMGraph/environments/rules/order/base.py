from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List

from pydantic import BaseModel
if TYPE_CHECKING:
    from LLMGraph.environments import BaseEnvironment
from . import order_registry

@order_registry.register("base")
class BaseOrder(BaseModel):
    rule_description:str=""
    type:str ="base"
    
    

    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        """Return the index of the next agent to speak"""
        pass

    def generate_deque(self, environment: BaseEnvironment) :
        """Return the index of the next agent to speak"""
        pass

    def requeue(self, environment: BaseEnvironment,tenant):
        """Return the index of the next agent to speak"""
        pass
        
    def are_all_deques_empty(self,environment):
        pass
    def reset(self) -> None:
        pass
