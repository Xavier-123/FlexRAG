"""
Knowledge base sub-package (indexing layer).

Exposes the FAISS-backed knowledge builder implementation.
"""

from flexrag.indexing.knowledge.faiss_knowledge import FaissKnowledgeBuilder

__all__ = ["FaissKnowledgeBuilder"]
