"""
Knowledge base sub-package.

Exposes the FAISS-backed knowledge base implementation that is used by
default when ``main.py`` is invoked.
"""

from flexrag.knowledge.faiss_knowledge import FaissKnowledgeBase

__all__ = ["FaissKnowledgeBase"]
