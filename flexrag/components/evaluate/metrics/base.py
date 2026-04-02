import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BaseMetric:
    metric_name: str = "base"

    def __init__(self) -> None:
        logger.debug(f"Loading {self.__class__.__name__}")

    async def calculate_metric_scores(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        计算该指标的聚合分数（pooled）以及每个样本的分数（example）。

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]
        """
        raise NotImplementedError(f"{self.__class__.__name__}.calculate_metric_scores must be implemented")
