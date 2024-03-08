
from LLMGraph.prompt.prompt import prompt_registry, prompt_default

from LLMGraph.prompt.prompt.base_prompt import BasePromptTemplate

@prompt_registry.register("forum")
class Forum_PromptTemplate(BasePromptTemplate):

    def __init__(self, **kwargs):
        template = kwargs.pop("template",
                              prompt_default.get("forum_template", ""))
        input_variables = kwargs.pop("input_variables",
                                     ["role_description",
                                      "community_name",
                                      "forum_comments"
                                      ])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
