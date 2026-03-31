import json
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

class _CustomReader(BaseReader):
    """自定义 JSON 读取器，专门处理 [{"idx": 0, "title": "...", "text": "..."}, ...] 格式。"""

    def load_data(self, file: str, extra_info: dict | None = None) -> list[Document]:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # 如果不是列表，返回空或抛出异常视具体业务而定
            raise ValueError(f"Expected a JSON array in {file}")

        docs = []
        for item in data:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "").strip()
            text = item.get("text") or item.get("context") or ""
            idx = item.get("idx")

            # 拼接 title 和 text 作为文档内容，有助于 Embedding 捕获完整语义
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if text:
                content_parts.append(text)

            content = "\n\n".join(content_parts)

            if not content:
                continue

            # 将 idx 存入 metadata，这样检索时可以顺带返回原始的 id
            metadata = {}
            if idx is not None:
                metadata["idx"] = idx

            docs.append(Document(text=content, metadata=metadata))

        return docs
