from __future__ import annotations

from typing import TYPE_CHECKING, List
from collections import deque
from . import order_registry as OrderRegistry
from .base import BaseOrder
import copy
import random





@OrderRegistry.register("kwaitlist")
class KWaitListOrder(BaseOrder):
    """
    k deferral waitlist order for tenants
    
    k : the tenant has k times to choose in one round
    if tenant.choose_times < k: tenant remain in waitlist
    
    waitlist_ratio : waitlist shortlisted ratio (default = 0.3 )
    
    """
    
    rule_description:str="""
Please note that before selecting a house, you need to follow these rules to queue up for house selection:\
The queue consists of the house-selection queue and the waiting queue. \
Being in the house-selection queue means you have a chance to select a house earlier. \
If you use up all your chances in the house selection queue, you will need to wait in the waiting queue to select a house.\
"""
    
    waitlist_ratio:float = 1.2 
    k:int = 2
    
    def __init__(self,**kargs) -> None:
        return super().__init__(type = "kwaitlist",
                                **kargs)
    
    def get_next_agent_idx(self, environment) ->dict: 
        # return queue_name:queue
        
        """Return the index of the next agent to speak"""
        waitlist_return = {}
        for queue_name, queue_info in environment.deque_dict.items():
            waitlist = copy.deepcopy(queue_info["waitlist"])
            queue_info["waitlist"] = []
            waitlist_return[queue_name] = waitlist
        return waitlist_return

    def generate_deque(self, environment):
        queues = copy.deepcopy(environment.tenant_manager.groups)
        
        for queue_name,queue_tenant_ids in queues.items():
            if queue_name not in environment.deque_dict.keys():
                environment.deque_dict[queue_name]={
                    "queue":queue_tenant_ids,
                    "waitlist":[]
                }
            else:
                for tenant_id in queue_tenant_ids:
                    if tenant_id not in environment.deque_dict[queue_name]["queue"] \
                        and tenant_id not in environment.deque_dict[queue_name]["waitlist"]: 
                        environment.deque_dict[queue_name]["queue"].append(tenant_id)
                
        
        return environment.deque_dict
        
        

    def requeue(self, environment, tenant):
        """re-queue"""
        if tenant.available:       
            queue_info = environment.deque_dict[tenant.queue_name]
            if tenant.round_choose_times >= self.k:
                tenant.finish_round()
                queue_info['queue'].append(tenant.id)
            else:
                queue_info["waitlist"].append(tenant.id)
                        
        
    def reset(self,environment) -> None:
        environment.deque_dict={}

    
    def enter_waitlist(self,environment):
        # 根据小区的数量改pool的大小
        pool_num_dict = environment.system.community_manager.get_pool_num() 
        for pool_name,pool_num in pool_num_dict.items():
            if pool_name not in environment.deque_dict.keys():
                continue
            
            if int(pool_num*self.waitlist_ratio)>(len(environment.deque_dict[pool_name]["queue"])+len(environment.deque_dict[pool_name]["waitlist"])):
                enter_tenant_ids = environment.deque_dict[pool_name]["queue"]
                for enter_tenant_id in enter_tenant_ids:
                    
                    environment.tenant_manager[enter_tenant_id].finish_round()
                    
                environment.deque_dict[pool_name]["waitlist"].extend(enter_tenant_ids)
                environment.deque_dict[pool_name]["queue"]=[]
            elif int(pool_num*self.waitlist_ratio)<len(environment.deque_dict[pool_name]["waitlist"]):
                continue
            else:
                enter_num=int(pool_num*self.waitlist_ratio)-len(environment.deque_dict[pool_name]["waitlist"])
                
                enter_tenant_ids = environment.deque_dict[pool_name]["queue"][:enter_num]
                for enter_tenant_id in enter_tenant_ids:
                    
                    environment.tenant_manager[enter_tenant_id].finish_round() # 从queue到waitlist中，相当于用完了waitlist中的机会了
                
                environment.deque_dict[pool_name]["waitlist"].extend(enter_tenant_ids)
                del environment.deque_dict[pool_name]["queue"][:enter_num]
                
    def are_all_deques_empty(self,environment) -> bool:     
        if environment.tenant_manager.available_tenant_num()<=0:
            return True
        
        self.enter_waitlist(environment)
        if all(len(queue_info["queue"])<=0 and \
            len(queue_info["waitlist"])<=0 for _,queue_info in environment.deque_dict.items()):
            return True
        return False

                  