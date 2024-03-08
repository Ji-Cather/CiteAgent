from LLMGraph.registry import Registry
order_registry = Registry(name="OrderRegistry")

from .base import BaseOrder
from .rent import RentOrder
from .priority import PriorityOrder
from .waitlist import WaitListOrder
from .kwaitlist import KWaitListOrder
