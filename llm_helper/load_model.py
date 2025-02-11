# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 10:27
# @Author  : zhanghaoxiang
# @File    : load_model.py
# @Software: PyCharm
#模型
#from pymilvus import MilvusClient

#client = MilvusClient("milvus_demo.db")
'''
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
model_name = "../LLM/Model/Modelbge-large-zh-v1.5/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
test=hf.embed_documents("今天天气怎么样")
vector=Chroma.from_texts(["今天天气怎么样，很好"],hf)
vector1=Chroma.from_texts(["今天天气因，很不好"],hf)
vector1=Chroma.from_texts(["今天天气多云，很好"],hf)
vector1=Chroma.from_texts(["今天天气热，很不好"],hf)
print(vector1.similarity_search("今天天气打雷",k=1))
#print(test)
'''
import numpy as np

def pearsonr(x, y):
    """
    计算x和y的Pearson相关系数
    """
    n = len(x)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    r = np.cov(x, y, ddof=1)[0, 1] / (sx * sy)
    return r
a=pearsonr( [51, 447, 548, 618],[249, 361, 368, 371])
print(a)

import numpy as np

vec1 = np.array([51, 447, 548, 618])
vec2 = np.array([79, 644, 516, 669])

simi12 = np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(simi12)

str="6_1_table.jpg"
ret=str.split("table")[0]
print(ret)