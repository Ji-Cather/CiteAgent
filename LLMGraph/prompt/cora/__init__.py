import yaml
cora_prompt_default = yaml.safe_load(open("LLMGraph/prompt/cora/prompt.yaml"))

from LLMGraph.registry import Registry
cora_prompt_registry = Registry(name="CoraPromptRegistry")

from .group_discuss import *