from LLMGraph.prompt.prompt import prompt_registry, prompt_default

from LLMGraph.prompt.prompt.base_prompt import BasePromptTemplate


# Set up a prompt template
@prompt_registry.register("choose_community")
class Choose_CommunityPromptTemplate(BasePromptTemplate):

    def __init__(self, **kwargs):
        template = kwargs.pop("template",
                              prompt_default.get("choose_community_template", ""))
        input_variables = kwargs.pop("input_variables",
                                     ["role_description",
                                      "choose_num",
                                      "house_info",
                                      "house_future_info"
                                      ])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)

