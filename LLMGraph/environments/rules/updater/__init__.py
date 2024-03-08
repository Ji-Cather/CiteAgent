from LLMGraph.registry import Registry
updater_registry = Registry(name="UpdaterRegistry")

from .base import BaseUpdater
from .article import ArticleUpdater