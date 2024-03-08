from langchain.tools import StructuredTool as BaseTool
from LLMGraph.message import Message
from typing import List
from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate


# Set up a prompt template
# 未用
@chat_prompt_registry.register("forum")
class ForumPromptTemplate(BaseChatPromptTemplate):
    # The list of tools available
    tools: List[BaseTool]
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("forum_template",""))
        input_variables = kwargs.pop("input_variables",
                    [
                     "task", 
                     "role_description",
                     "memory",
                     "act_time",
                     "agent_scratchpad"])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        # task = kwargs.get("task","publish")

        
        
        formatted = self.template.format(**kwargs)
        
        return [Message(content=formatted,
                        message_type="publish")]