from LLMGraph.message import Message
from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate
from langchain.schema import HumanMessage

# Set up a prompt template
@chat_prompt_registry.register("choose")
class ChoosePromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("choose_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["task", 
                     "role_description",
                     "house_info",
                     "thought_type",
                     "thought_hint",
                     "choose_type",
                     "memory",
                     "agent_scratchpad"])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        # task = kwargs.get("task","choose")
        # message_type= "choose"
        # if 'You need to choose one type of communities.' == task:
        #     message_type = "community"
        # elif 'You need to choose one type of houses.' == task:
        #     message_type = "house_type"
        # elif 'You need to choose one house.' == task:
        #     message_type = "house"
            
        formatted = self.template.format(**kwargs)
    
        # return [Message(content=formatted,
        #                 message_type=message_type)]        
        return [HumanMessage(content=formatted)]