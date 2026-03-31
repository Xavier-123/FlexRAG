import asyncio
from abc import ABC, abstractmethod


class BaseQueryOptimizer(ABC):
    """Strategy interface for rewriting retrieval queries during iteration."""

    @abstractmethod
    async def run(
        self,
        original_query: str,
        accumulated_context: list[str],
        missing_info: str,
        previous_query: str = "",
    ) -> dict:
        """
        Args:
            original_query: The user's original question.
            accumulated_context: Context collected in previous iterations.
            missing_info: Feedback on what information is still missing.
            previous_query: The query used in the previous iteration.
        """