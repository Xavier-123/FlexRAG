from abc import ABC, abstractmethod
from typing import Any
from flexrag.common.schema import Document


class BasePostRetrieval(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int,
    ) -> Any:
        pass