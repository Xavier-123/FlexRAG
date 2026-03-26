import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Callable

from flexrag.evaluate.utils.eval_utils import normalize_answer
from flexrag.evaluate.metrics.base import BaseMetric


class CharF1Score(BaseMetric):
    metric_name: str = "char_f1_score"

    def __init__(self):
        super().__init__()

    async def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        aggregation_fn: Callable = np.max,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results


if __name__ == '__main__':
    common = Counter("hi hello") & Counter("hells hi")
    print(common)