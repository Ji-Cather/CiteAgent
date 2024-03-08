from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Dict

from langchain.schema import AgentAction
from pydantic import BaseModel


from LLMGraph.message import Message

from . import updater_registry as UpdaterRegistry

from .base import BaseUpdater


@UpdaterRegistry.register("article")
class ArticleUpdater(BaseUpdater):
    """
    The basic version of updater.
    The messages will be seen by all the receiver specified in the message.
    """
    def post_messages(self, 
                      environment):
        for agent_group in environment.agent_groups:
            for agent in agent_group["authors"]:
                post_messages = agent.post_messages()
                self.post_social_network(post_messages,
                                         agent_group)
                
    def post_social_network(self,
                            post_messages,
                            agent_group):
        
        for message in post_messages:
            for r_id in message.receivers:
                for agent_r in agent_group["authors"]:
                    if r_id == agent_r.name:
                        agent_r.receive_messages(messages=[message])
                    
    def reset(self):
        pass