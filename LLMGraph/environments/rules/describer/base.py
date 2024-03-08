from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from pydantic import BaseModel

from . import describer_registry as DescriberRegistry

if TYPE_CHECKING:
    from LLMGraph.environments import BaseEnvironment


@DescriberRegistry.register("base")
class BaseDescriber(BaseModel):
    rule_description:str=""
    def reset(self) -> None:
        pass
