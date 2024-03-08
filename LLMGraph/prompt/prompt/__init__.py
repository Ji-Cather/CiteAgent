import yaml
prompt_default = yaml.safe_load(open("LLMGraph/prompt/prompt/prompt.yaml"))

from LLMGraph.registry import Registry
prompt_registry = Registry(name="PromptRegistry")
from LLMGraph.prompt.prompt.base_prompt import BasePromptTemplate
from LLMGraph.prompt.prompt.choose_house import Choose_HousePromptTemplate
from LLMGraph.prompt.prompt.choose_community import Choose_CommunityPromptTemplate
from LLMGraph.prompt.prompt.comment import CommentPromptTemplate
from LLMGraph.prompt.prompt.forum import Forum_PromptTemplate
from LLMGraph.prompt.prompt.correct import Correct_Choose_House_PromptTemplate,Correct_Choose_Community_PromptTemplate