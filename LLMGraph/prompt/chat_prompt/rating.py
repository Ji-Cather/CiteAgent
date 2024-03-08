from LLMGraph.message import Message
from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate
from langchain.schema import HumanMessage

# Set up a prompt template
@chat_prompt_registry.register("choose_rating")
class ChooseRatingPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("choose_rating_template",""))

        input_variables = kwargs.pop("input_variables",
                    [
                     "role_description",
                     "house_info",
                     "memory",
                     "available_house_index",
                     "agent_scratchpad"])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:    
        formatted = self.template.format(**kwargs)
    
        # return [Message(content=formatted,
        #                 message_type=message_type)]
        return [HumanMessage(content=formatted)]