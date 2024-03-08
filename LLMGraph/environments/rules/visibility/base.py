from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from LLMGraph.environments import BaseEnvironment
from . import visibility_registry

@visibility_registry.register("base")
class BaseVisibility(BaseModel):
    rule_description:str=""
    def filter_community(self, tenant,community_list):
        
        return community_list
    
    def filter_housetype(self, tenant,housetype_list):
        
        return housetype_list
    def reset(self):
        pass