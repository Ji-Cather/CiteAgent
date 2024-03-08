from LLMGraph.prompt.prompt import prompt_registry, prompt_default

from LLMGraph.prompt.prompt.base_prompt import BasePromptTemplate

@prompt_registry.register("correct_choose_community")
class Correct_Choose_Community_PromptTemplate(BasePromptTemplate):

    def __init__(self, **kwargs):
        template = kwargs.pop("template",
                              prompt_default.get("correct_format_choose_community", ""))
        input_variables = kwargs.pop("input_variables",
                                     ["response",
                                      ])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)

@prompt_registry.register("correct_choose_house")
class Correct_Choose_House_PromptTemplate(BasePromptTemplate):

    def __init__(self, **kwargs):
        template = kwargs.pop("template",
                              prompt_default.get("correct_format_choose_house", ""))
        input_variables = kwargs.pop("input_variables",
                                     ["response",
                                      ])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)