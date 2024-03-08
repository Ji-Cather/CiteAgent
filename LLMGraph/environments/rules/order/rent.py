from __future__ import annotations

from typing import TYPE_CHECKING, List
from collections import deque
from . import order_registry as OrderRegistry
from .base import BaseOrder
import re
import random





@OrderRegistry.register("rent")
class RentOrder(BaseOrder):
    """
    Order for rent.
    random(agents.tenants) (one agent at a time)
    """
    rule_description:str=""
    
    def __init__(self,**kargs) -> None:
        return super().__init__(type = "rent",
                                **kargs)
    
    def get_next_agent_idx(self, environment) :
        """Return the index of the next agent to speak"""
        result=[]
        for _,deque in environment.deque_dict.items():
            if len(deque)>0:
                result.append(deque.popleft())
        return result

    def generate_deque(self, environment):
        tenantlist=list(environment.tenant_manager.data.values())
        # random.shuffle(tenantlist) # 测试
        deque_list = deque(tenantlist) 
        environment.deque_dict["random_queue"] = deque_list
        return environment.deque_dict

    def requeue(self, environment,tenant):
        if (tenant.choose_times>=tenant.max_choose):
            return
        """re-queue"""
        environment.deque_dict["random_queue"].append(tenant)
        
    def reset(self,environment) -> None:
        environment.deque_dict["random_queue"].clear()

    def are_all_deques_empty(self,environment) -> bool:
        return all(len(d) == 0 for d in environment.deque_dict.values())