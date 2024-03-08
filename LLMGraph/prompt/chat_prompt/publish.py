from langchain.tools import StructuredTool as BaseTool
from langchain.schema import HumanMessage
from LLMGraph.message import Message
from typing import List
from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate


# Set up a prompt template
@chat_prompt_registry.register("publish_forum")
class PublishPromptTemplate(BaseChatPromptTemplate):
    # The list of tools available

    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("publish_forum_template",""))
        input_variables = kwargs.pop("input_variables",
                    [
                     "role_description",
                     "plan",
                     "memory",
                     "agent_scratchpad",
                     "community_ids"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:

        formatted = self.template.format(**kwargs)
        
        # return [Message(content=formatted,
        #                 message_type="publish")]
        return [HumanMessage(content=formatted)]
    
    
    
@chat_prompt_registry.register("publish_forum_plan")
class PublishPromptPlanTemplate(BaseChatPromptTemplate):
    # The list of tools available

    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("publish_forum_plan_template",""))
        input_variables = kwargs.pop("input_variables",
                    [
                     "concise_role_description", 
                     "memory",
                     "system_competiveness_description",
                     "personality",
                     "agent_scratchpad",
                     "goal",
                     "respond_format"
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:
        
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]