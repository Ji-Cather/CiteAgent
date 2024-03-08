from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from .base import BaseVisibility
if TYPE_CHECKING:
    from LLMGraph.environments import BaseEnvironment
from . import visibility_registry
import copy

@visibility_registry.register("rent")
class RentVisibility(BaseVisibility):
    rule_description:str = """""" 
    
# 第一个版本的rule
    available_housetype:dict = {
            "1":["small_house"],
            "2":["small_house","middle_house"],
            "3":["large_house","middle_house"]
        }
    
    def filter_community(self, tenant, community_list, house_type=None):
        community_list_copy = copy.deepcopy(community_list.copy())
        """Update the set of visible agents for the agent"""
        community_filter_list = []
        
        tenant_group = tenant.family_num if tenant.family_num < 3 else 3
        
        available_house_types = self.available_housetype[str(tenant_group)] if house_type is None \
            else [house_type]
            
        for community_info in community_list_copy:
            if  community_info["sum_remain_num"] >  0 :
                for housetype,housetype_att in list(community_info.items()):
                    if isinstance(housetype_att, dict)  and housetype not in available_house_types:
                            del community_info[housetype]
                for housetype,housetype_att in list(community_info.items()):
                    if isinstance(housetype_att, dict):
                            community_filter_list.append(community_info)
                            break
        return community_filter_list
    
    # def filter_housetype(self, tenant, housetype_list):
    #     housetype_list_copy = housetype_list
    #     housetype_filter_list=[]
    #     if tenant.family_num<=0:
    #         return []
    #     elif tenant.family_num==1:
    #         for housetype in housetype_list_copy:
    #             if  housetype in self.available_housetype["1"]:
    #                 housetype_filter_list.append(housetype)
    #     elif tenant.family_num==2:
    #         for housetype in housetype_list_copy:
    #             if  housetype in self.available_housetype["2"]:
    #                 housetype_filter_list.append(housetype)
    #     elif tenant.family_num>=3:
    #         for housetype in housetype_list_copy:
    #             if  housetype in self.available_housetype["3"]:
    #                 housetype_filter_list.append(housetype)
    #     return housetype_filter_list

    def reset(self):
        pass