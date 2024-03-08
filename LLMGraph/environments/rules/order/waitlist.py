from __future__ import annotations

from typing import TYPE_CHECKING, List
from collections import deque
from . import order_registry as OrderRegistry
from .base import BaseOrder
import copy
import random





@OrderRegistry.register("waitlist")
class WaitListOrder(BaseOrder):
    """
    k deferral waitlist order for tenants
    
    k (tenant.max_choose): the tenant has k times to choose
    if tenant.choose_times < tenant.max_choose: tenant remain in waitlist
    
    waitlist_ratio : waitlist shortlisted ratio (default = 0.3 )
    
    """
    
    rule_description:str=""
    waitlist_ratio:float= 0.5 
    
    def __init__(self,**kargs) -> None:
        return super().__init__(type = "waitlist",
                                **kargs)
    
    def get_next_agent_idx(self, environment) :
        """Return the index of the next agent to speak"""
        waitlist = environment.deque_dict["waitlist"]
        if (waitlist["cur_num"] < waitlist["num"]): # 需要补充wailist
            add_num = waitlist["num"] - waitlist["cur_num"]
        
            queues_all = []
            for group_id, group_queue in environment.deque_dict.items():
                if (group_id == "waitlist"):
                    continue
                queues_all.extend(group_queue["queue"])
                
            queues_all_p = []
            queues_all_np = []
            for tenant in queues_all:
                if all(not value for value in tenant.priority_item.values()):
                    queues_all_p.append(tenant)
                else:
                    queues_all_np.append(tenant)
                    
            random.shuffle(queues_all_p)
            random.shuffle(queues_all_np)
            queues_all =[*queues_all_p,*queues_all_np]
            add_tenants = queues_all[:add_num] if add_num < len(queues_all) else queues_all
            
            waitlist["waitlist"].extend(add_tenants)
            
            waitlist["cur_num"] += len(add_tenants)
            for add_tenant in add_tenants:
                for group_id, group_queue in environment.deque_dict.items():
                    if (group_id == "waitlist"):
                        continue
                    if add_tenant.id in group_queue["tenant_ids"]:
                        for idx,tenant_queue in enumerate(group_queue["queue"]):
                            if tenant_queue.id == add_tenant.id:
                                group_queue["queue"].pop(idx)
                                break

            
        waitlist_return = waitlist["waitlist"]
        waitlist["waitlist"] = []
        waitlist["cur_num"] = 0
        return waitlist_return

    def generate_deque(self, environment):
        waitlist_queue = []
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
            group_queue = [*priority_queue,*non_priority_queue]
            environment.deque_dict[group_id]={
                "queue":group_queue,
                "group_num":len(tenant_ids),
                "tenant_ids": tenant_ids,
                } # group_id 号group队列内 具有优先资格 and 无优先资格的 tenant 队列，以及总人数
            
            len_group = int(len(tenant_ids)* self.waitlist_ratio)
            waitlist_queue.extend(group_queue[:len_group])
            environment.deque_dict[group_id]["queue"] = group_queue[len_group:]
        
        environment.deque_dict["waitlist"] = {
            "num":len(waitlist_queue),
            "waitlist":waitlist_queue,
            "cur_num":len(waitlist_queue)
        }
        
        return environment.deque_dict

    def requeue(self, environment,tenant):
        tenant_id=tenant.id
        """re-queue"""
        round_choose = tenant.choose_times % (tenant.max_choose)
        if (round_choose == 0): # choose 了max_choose次，该回原来的group队列了
            for group_id, group_queue in environment.deque_dict.items():
                if group_id == "waitlist":
                    continue
                elif tenant_id in group_queue["tenant_ids"]:
                    group_queue["queue"].append(environment.tenant_manager[tenant_id])
                    break
        else:
            environment.deque_dict["waitlist"]["waitlist"].append(tenant)
            environment.deque_dict["waitlist"]["cur_num"]+=1
        
    def reset(self,environment) -> None:
        for group_id, group_queue in environment.deque_dict.items():
                if group_id == "waitlist":
                    group_queue["waitlist"]=[]
                    group_queue["cur_num"]=0
                else:
                    group_queue["queue"]=[]
                    group_queue["group_num"]=0
                    group_queue["tenant_ids"]=[]

    def are_all_deques_empty(self,environment) -> bool:
        
        for group_id, group_queue in environment.deque_dict.items():
                if group_id == "waitlist":
                    if group_queue["cur_num"]>0:
                        return False
                elif len(group_queue["queue"]) >0:
                    return False
        
        return True