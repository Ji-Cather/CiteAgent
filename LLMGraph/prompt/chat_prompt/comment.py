from langchain.schema import HumanMessage
from LLMGraph.message import Message

from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate


# Set up a prompt template
@chat_prompt_registry.register("comment")
class CommentPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("comment_template",""))
        input_variables = kwargs.pop("input_variables",
                    ["message_type",
                     "role_description",
                     "house_info",
                     "comment_type",
                     "thought_type",
                     "memory",
                     "agent_scratchpad"])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        message_type = kwargs.pop("message_type","community")
        formatted = self.template.format(**kwargs)
    
        # return [Message(content=formatted,
        #                 message_type=message_type)]
        return [HumanMessage(content=formatted)]