from LLMGraph.message import Message
from LLMGraph.prompt.cora import cora_prompt_default,cora_prompt_registry

from LLMGraph.prompt.cora.base_chat_prompt import BaseChatPromptTemplate
    
    
@cora_prompt_registry.register("group_discuss")
class GroupDiscussPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             cora_prompt_default.get("group_discuss_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["character_1",
                     "character_2",
                     "past_context",
                     "cur_context",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    

@cora_prompt_registry.register("choose_researcher")
class ChooseResearcherPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             cora_prompt_default.get("choose_researcher_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "research_topic",
                     "past_context",
                     "researcher",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
        
@cora_prompt_registry.register("get_topic")
class GetTopicPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             cora_prompt_default.get("get_topic",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "research_topic",
                     "past_context",
                     "researcher",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)