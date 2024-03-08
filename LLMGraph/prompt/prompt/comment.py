
from LLMGraph.prompt.prompt import prompt_registry, prompt_default

from LLMGraph.prompt.prompt.base_prompt import BasePromptTemplate


# Set up a prompt template
@prompt_registry.register("comment")
class CommentPromptTemplate(BasePromptTemplate):

    def __init__(self, **kwargs):
        template = kwargs.pop("template",
                              prompt_default.get("comment_template", ""))
        input_variables = kwargs.pop("input_variables",
                                     ["role_description",
                                      "community_description",
                                      "house_description",
                                      "potential_information_house"
                                      ])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
