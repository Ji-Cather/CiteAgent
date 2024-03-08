from LLMGraph.message import Message
from LLMGraph.prompt.chat_prompt import chat_prompt_default,chat_prompt_registry

from LLMGraph.prompt.chat_prompt.base_chat_prompt import BaseChatPromptTemplate
from langchain.schema import HumanMessage

# Set up a prompt template
@chat_prompt_registry.register("action_plan")
class ActionPlanPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             chat_prompt_default.get("action_plan_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description", 
                     "actions",
                     "action_names",
                     "memory",
                     "history"]) #agent_scratchpad 会有bug，暂时用history代替
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
            
        formatted = self.template.format(**kwargs)
    
        # return [Message(content=formatted,
        #                 message_type=message_type)]
        return [HumanMessage(content=formatted)]