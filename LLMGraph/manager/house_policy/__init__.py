from LLMGraph.registry import Registry

house_patch_registry = Registry(name="HousePatchPolicyRegistry")

from .base import BaseHousePatchPolicy
