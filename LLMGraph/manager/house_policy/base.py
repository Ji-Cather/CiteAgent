from . import house_patch_registry
from pydantic import BaseModel
from abc import abstractmethod
from typing import List
import random
import numpy as np

@house_patch_registry.register("base")
class BaseHousePatchPolicy(BaseModel):
    
    type = "base"
   
    async def group(self,
                queue_house_ids,
                queue_num:int,
                queue_names:List[str]): # tenant_ids 为最新添加的一批tenant
        # return group_id
        random.shuffle(queue_house_ids)
        num_groups = queue_num
        n_per_group = len(queue_house_ids) // num_groups
        end_p = n_per_group*num_groups
        if end_p == len(queue_house_ids):
            end_p = -1
            groups = np.array(queue_house_ids).reshape(num_groups, n_per_group)
        else:
            groups = np.array(queue_house_ids[:end_p]).reshape(num_groups, n_per_group)
            
        groups = groups.tolist()
        if (end_p != -1):
            groups[-1].extend(queue_house_ids[end_p:])
            
        queue_group_h_ids = {}
        for idx,queue_name in enumerate(queue_names):
            queue_group_h_ids[queue_name] = groups[idx]
        return queue_group_h_ids
    