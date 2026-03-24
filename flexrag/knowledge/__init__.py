"""
Knowledge base sub-package.

Exposes the FAISS-backed knowledge builder implementation that is used by
default when ``main.py`` is invoked.
"""

from flexrag.knowledge.faiss_knowledge import FaissKnowledgeBuilder

__all__ = ["FaissKnowledgeBuilder"]
