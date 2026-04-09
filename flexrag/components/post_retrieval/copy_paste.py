from typing import Any
from CopyPasteLLM import CopyPasteClient

from flexrag.common.schema import Document
from flexrag.components.post_retrieval.base import BasePostRetrieval


class CopyPasteRetrieval(BasePostRetrieval):
    def __init__(self, model: str, base_url: str, api_key: str = None, pipeline=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if pipeline is None:
            pipeline = ["cp-refine"]
        self.pipeline = pipeline  # Options: cp-order, cp-link, cp-refine

        self.client = CopyPasteClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
            verbose=True
        )

    def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int
    ) -> Any:
        # Simply return the retrieved documents without modification
        context = "\n\n".join(doc.text for doc in documents)

        response_order = self.client.responses.create(
            context=''.join(context),
            query=query,
            pipeline="cp-order"  # Options: cp-order, cp-link, cp-refine
        )

        response_link = self.client.responses.create(
            context=''.join(context),
            query=query,
            pipeline="cp-link"  # Options: cp-order, cp-link, cp-refine
        )


        response_refine = self.client.responses.create(
            context=''.join(context),
            query=query,
            pipeline="cp-refine"  # Options: cp-order, cp-link, cp-refine
        )

        # # Access results
        # print(response.content)  # Generated text
        # print(response.extractiveness_score)  # 0.0 to 1.0 (higher = more extractive)
        #
        # # Visualize extractiveness
        # print(response.render_heatmap(documents))
        #
        # # Get extracted fragments
        # fragments = response.get_fragments(documents, min_length=2)

        return
