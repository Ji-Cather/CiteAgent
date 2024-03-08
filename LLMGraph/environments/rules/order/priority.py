from __future__ import annotations

from typing import TYPE_CHECKING, List
from collections import deque
from . import order_registry as OrderRegistry
from .base import BaseOrder
import re
import random





@OrderRegistry.register("priority")
class PriorityOrder(BaseOrder):
    """
    Order for rent.
    random(agents.tenants) (one agent at a time)
    """
    rule_description:str="""
Please note that before selecting a room, you need to select the room type queue and follow the following rules:
    1. If you are single, the optional room type queue includes small room types.
    2. If you belong to a family with a spouse or children under the age of 10, the available room types include small room and medium room.
    3. If you belong to a family with a spouse or children over 10 years old (with a family size of 2 or 3 people), \
the optional room type queue includes medium room type and large room type

Within each housing selection queue, the following families have priority: urban low-income families, low-income families,\
families with major illnesses or surgeries, severely disabled families, and families with special family planning difficulties.
"""


    def __init__(self,**kargs) -> None:
        return super().__init__(type = "priority",
                                **kargs)
    
    def get_next_agent_idx(self, environment) -> dict:
        """Return the index of the next agent to speak"""
        # result=[]
        # for _,deque in environment.deque_dict.items():
        #     if len(deque)>0:
        #         result.append(deque.popleft()) # 这个如何体现优先？
        # return result
        
        
        # 我觉得优先是priority_queue 都选完，才轮到non_priority_queue
        result = [] # group_id:tenant_id
        
        for group_id, queues in environment.deque_dict.items():
            if len(queues["priority_queue"]) > 0:
                result.append(queues["priority_queue"].popleft())
                environment.deque_dict[group_id]["group_num"] -= 1
            elif len(queues["non_priority_queue"]) > 0:
                result.append(queues["non_priority_queue"].popleft())
                environment.deque_dict[group_id]["group_num"] -= 1
            

        return result

    def generate_deque(self, environment):
        
        # for tenant in environment.tenant_manager.data.values():
        #     if all(not value for value in tenant.priority_item.values()):
        #         non_priority_queue.append(tenant)
        #     else:
        #         priority_queue.append(tenant)
        
        for group_id,tenant_ids in environment.tenant_manager.groups.items():
            priority_queue = []
            non_priority_queue = []
            
            for tenant_id in tenant_ids:
                tenant = environment.tenant_manager[tenant_id]
                if all(not value for value in tenant.priority_item.values()):
                    non_priority_queue.append(tenant)
                else:
                    priority_queue.append(tenant)
                
        
            # 将这两个队列添加到 environment 的 deque_dict 中
            random.shuffle(priority_queue)
            random.shuffle(non_priority_queue)
            environment.deque_dict[group_id]={
                "priority_queue":deque(priority_queue),
                "non_priority_queue":deque(non_priority_queue),
                "group_num":len(tenant_ids)
                } # group_id 号group队列内 具有优先资格 and 无优先资格的 tenant 队列，以及总人数
            
        return environment.deque_dict

    def requeue(self, environment,tenant):
        if (tenant.choose_times >= tenant.max_choose):
            return
        tenant_id = tenant.id 
        for group_id,tenant_ids in environment.tenant_manager.groups.items():
            if tenant_id in tenant_ids:
                if all(not value for value in tenant.priority_item.values()):
                    environment.deque_dict[group_id]["priority_queue"].append(tenant)
                else:
                    environment.deque_dict[group_id]["non_priority_queue"].append(tenant)
                return

    def reset(self,environment) -> None:
        for group_id, queues in environment.deque_dict.items():
            for queue in queues:    
                queue.clear()

        
        
    def are_all_deques_empty(self,environment) -> bool:
        return all(queue["group_num"] == 0 for queue in environment.deque_dict.values())