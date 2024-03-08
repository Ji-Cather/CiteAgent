from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from langchain.schema import AgentAction
from pydantic import BaseModel


from LLMGraph.message import Message

from . import updater_registry as UpdaterRegistry



@UpdaterRegistry.register("base")
class BaseUpdater(BaseModel):
    """
    The basic version of updater.
    The messages will be seen by all the receiver specified in the message.
    """
    def post_messages(self, **kwargs):
        pass

        
    def reset(self):
        pass