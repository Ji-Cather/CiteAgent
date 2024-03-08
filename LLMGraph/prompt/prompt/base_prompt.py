from langchain.prompts import PromptTemplate
from LLMGraph.prompt.prompt import prompt_registry


# Set up a prompt template
@prompt_registry.register("base")
class BasePromptTemplate(PromptTemplate):
    # The template to use
    template: str

