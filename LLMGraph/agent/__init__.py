from LLMGraph.registry import Registry
agent_registry = Registry(name="AgentRegistry")
manager_agent_registry = Registry(name="ManagerAgentRegistry")


from .article import ArticleAgent,ArticleManagerAgent
from .tool_agent import ToolAgent