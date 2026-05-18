"""
framework.loader — entity loading and declarative graph construction.
"""

from framework.loader.entity_loader import EntityLoader
from framework.loader.graph_builder import _DEFAULT, _get_state_schemas

__all__ = [
    "EntityLoader",
    "_DEFAULT",
    "_get_state_schemas",
]
