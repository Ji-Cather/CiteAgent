from LLMGraph.registry import Registry

output_parser_registry = Registry(name="OutputParserRegistry")
article_output_parser_registry = Registry(name="ArticleOutputParserRegistry")
control_output_parser_registry = Registry(name="ControlOutputParserRegistry")
from .article import *
from .control import *