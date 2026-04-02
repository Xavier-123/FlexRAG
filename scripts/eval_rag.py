import argparse
import asyncio
import json
from typing import Any, Dict, List

from flexrag.components.evaluate.metrics.em import ExactMatch
from flexrag.components.evaluate.metrics.f1 import CharF1Score
from flexrag.components.evaluate.metrics.recall_k import NonLLMContextRecall

InputItem = Dict[str, Any]


def _parse_int_list(s: str) -> List[int]:
    # 支持形如 "1,3,5" / "1 3 5"
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="eval_results.json",
        help="eval_results.json path (默认读取当前目录下 eval_results.json)",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,5,10,20",
        help="Recall@k 的 k 列表，例如：1,3,5",
    )
    args = parser.parse_args()

    input_path = args.input
    k_list = _parse_int_list(args.k_list)

    with open(input_path, "r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("eval_results.json must be a JSON array")

    results: List[InputItem] = raw
    if not results:
        raise ValueError("eval_results.json is empty")

    # 评估输入数据结构适配：
    # - EM/F1: gold_answers: List[List[str]]; predicted_answers: List[str]
    # - Recall@k: gold_docs/retrieved_docs: List[List[str]]
    #
    # 这里 eval_results.json 只有 `evidence` 字段，没有显式区分 gold_docs 与 retrieved_docs，
    # 因此暂时采用 evidence 同时作为两者。
    gold_answers: List[List[str]] = []
    predicted_answers: List[str] = []
    gold_docs: List[List[str]] = []

    for idx, item in enumerate(results):
        expected = item.get("expected")
        generated_answer = item.get("generated_answer")
        evidence = item.get("evidence", [])

        if expected is None or generated_answer is None:
            raise ValueError(f"Missing `expected` or `generated_answer` at index={idx}")
        if not isinstance(evidence, list):
            raise ValueError(f"`evidence` must be a list at index={idx}")

        gold_answers.append([str(expected)])
        predicted_answers.append(str(generated_answer))
        gold_docs.append([str(e) for e in evidence])

    retrieved_docs = gold_docs

    em_metric = ExactMatch()
    f1_metric = CharF1Score()
    recall_metric = NonLLMContextRecall()

    (em_pooled, em_example), (f1_pooled, f1_example), (recall_pooled, recall_example) = await asyncio.gather(
        em_metric.calculate_metric_scores(gold_answers, predicted_answers),
        f1_metric.calculate_metric_scores(gold_answers, predicted_answers),
        recall_metric.calculate_metric_scores(gold_docs, retrieved_docs, k_list=k_list),
    )

    print("overall scores:")
    print(f"  EM: {em_pooled}")
    print(f"  F1: {f1_pooled}")
    print(f"  Recall@k: {recall_pooled}")

    print("\nper-example scores (first 5):")
    for i in range(min(5, len(results))):
        q = results[i].get("question", f"index={i}")
        print(f"\n- {q}")
        print(f"  EM: {em_example[i]}")
        print(f"  F1: {f1_example[i]}")
        print(f"  Recall@k: {recall_example[i]}")


if __name__ == "__main__":
    asyncio.run(main())