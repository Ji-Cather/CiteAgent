from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel
import random
from LLMGraph.environments.rules.order import BaseOrder, order_registry
from LLMGraph.environments.rules.visibility import BaseVisibility, visibility_registry
from LLMGraph.environments.rules.updater import BaseUpdater, updater_registry
from LLMGraph.environments.rules.describer import BaseDescriber, describer_registry


if TYPE_CHECKING:
    from LLMGraph.environments.base import BaseEnvironment

from LLMGraph.message import Message


class Rule(BaseModel):
    """
    Rule for the environment. It controls the speaking order of the agents 
    and maintain the set of visible agents for each agent.
    """
    order: BaseOrder
    visibility: BaseVisibility
    updater: BaseUpdater
    describer: BaseDescriber

    def __init__(self, 
                 order_config,
                 updater_config,
                 visibility_config,
                 describer_config
                 ):
        order = order_registry.build(**order_config)
        updater = updater_registry.build(**updater_config)
        visibility = visibility_registry.build(**visibility_config)
        describer = describer_registry.build(**describer_config)
        super().__init__(order=order,
                         updater=updater,
                         visibility=visibility,
                         describer=describer)

    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        """Return the index of the next agent to speak"""
        return self.order.get_next_agent_idx(environment)

    def generate_deque(self, environment):
        return self.order.generate_deque(environment)

    def requeue(self, environment, tenant):
        self.order.requeue(environment,tenant)

    def reset(self, environment: BaseEnvironment) -> None:
        self.order.reset()
        
    def are_all_deques_empty(self, environment: BaseEnvironment) -> None:
        return self.order.are_all_deques_empty(environment)
    
    def post_messages(self,**kargs):
        return self.updater.post_messages(**kargs)
    
    def filter_community(self, **kwargs):
        """Update the set of visible agents for the agent"""
        return self.visibility.filter_community(**kwargs)    
        
    # def filter_housetype(self, tenant,housetype_list):
    #     """Update the set of visible agents for the agent"""
    #     return self.visibility.filter_housetype(tenant,housetype_list)
    
    def rule_description(self):
        return self.order.rule_description+self.visibility.rule_description