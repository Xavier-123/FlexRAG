from llama_index.core import Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# 替换为你的远程服务地址，通常以 /v1 结尾
API_BASE = "http://127.0.0.1:19002/v1"
API_KEY = "sk-1234567890"  # 如果没有设置鉴权，可以填 "EMPTY" 或随意填
MODEL_NAME = "Qwen3-Embedding-0.6B"  # 比如 "bge-m3" 等

embed_model = OpenAILikeEmbedding(
    model_name=MODEL_NAME,
    api_base=API_BASE,
    api_key=API_KEY,
    # embed_batch_size=10,
)

# 设置为全局 Embedding 模型
Settings.embed_model = embed_model

# 测试
embeddings = embed_model.get_text_embedding("测试一下远程模型是否连通")
print("======================================")
print(len(embeddings))


# import requests
# # 测试路径
# url = "http://127.0.0.1:19002/v1/embeddings"
#
# payload = {
#     "input": "测试一下",
#     "model": "Qwen3-Embedding-0.6B"
# }
# headers = {"Authorization": "Bearer sk-1234567890"}
#
# response = requests.post(url, json=payload, headers=headers)
# print("状态码:", response.status_code)
# print("返回值:", response.text)