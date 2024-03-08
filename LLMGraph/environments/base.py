from typing import Any, Dict, List

from pydantic import BaseModel

from LLMGraph.environments.rules.base import Rule
from LLMGraph.message import Message
from LLMGraph.environments.rules.base import Rule

from pydantic import BaseModel

from . import env_registry as EnvironmentRegistry


@EnvironmentRegistry.register("base")
class BaseEnvironment(BaseModel):
    rule: Rule
    max_turns: int = 10
    cnt_turn: int = 0
    rule_params: Dict = {}

    def __init__(self, rule, **kwargs):
        rule_config = rule
        order_config = rule_config.get('order', {'type': 'rent'})
        updater_config = rule_config.get('updater', {'type': 'base'})
        visibility_config = rule_config.get('visibility', {'type': 'base'})
        describer_config = rule_config.get('describer', {'type': 'base'})

        rule = Rule(order_config,
                    updater_config,
                    visibility_config,
                    describer_config)
        
        super().__init__(rule=rule, **kwargs)