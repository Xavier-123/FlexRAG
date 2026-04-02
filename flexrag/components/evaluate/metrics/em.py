import logging
import numpy as np
from typing import List, Dict, Tuple, Callable

from flexrag.components.evaluate.utils.eval_utils import normalize_answer
from flexrag.components.evaluate.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class ExactMatch(BaseMetric):
    '''ExactMatch指标检查响应是否与参考文本完全相同。在需要确保生成的响应逐字匹配预期输出的场景中，它很有用。'''
    metric_name: str = "exact_match"

    def __init__(self):
        super().__init__()

    async def calculate_metric_scores(
            self,
            gold_answers: List[List[str]],
            predicted_answers: List[str],
            aggregation_fn: Callable = np.max,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results
