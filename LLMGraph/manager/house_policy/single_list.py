from .base import BaseHousePatchPolicy


class SingleListPatchPolicy(BaseHousePatchPolicy):
    async def group(self,
                queue_house_ids,
                queue_num:int,
                queue_names:List[str]): # tenant_ids 为最新添加的一批tenant
        