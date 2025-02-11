# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 9:26
# @Author  : zhanghaoxiang
# @File    : embeddding_Text.py
# @Software: PyCharm
import json
import requests
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
class embedding_Text:
    def __init__(self, post_url:str, post_data:str):
        self.post_url = post_url
        self.post_data=post_data
    def request_data(self)->str:
        '''
        将传入的字符串，配置成对应的josn格式
        后续会采用langchain的形式进行整改
        :return:
        '''
        modeName="bge-large-zh"
        data={
            "model":modeName,
            "engine": "string",
            "input":self.post_data,
            "user": 'zhx',
            "encoding_format": "string"
        }
        return json.dumps(data)
    def request(self)->(int, str):
        try:
            # 发送 POST 请求
            response = requests.post(self.post_url, self.request_data())
            # 检查响应状态码
            if response.status_code == 200:
                print('Request was successful.')
                print('Response JSON:', response.json())
                return 200,response.json()
            else:
                print(f'Request failed with status code {response.status_code}')
                return response.status_code, "Internal Server Error"
        except requests.exceptions.RequestException as e:
                print('An error occurred:', e)
                #自定义的错误码
                return 501, e
    def responsePares(self,code:int,response:str)->(int,dict):
        try:
            if code == 200:
                data = {}
                data["index"] = response["data"][0]["index"]
                data["vector"] = response["data"][0]["embedding"]
                data['tokens'] = response["usage"]["total_tokens"]
                return 201, data
            else:
                return 202,None
        except Exception as e:
                return 203, None
if __name__ == "__main__":
    '''
    测试代码
    '''
    model=embedding_Text(post_url="http://100.120.0.26:8000/v1/embeddings?model_name=bge-large-zh", post_data="今天是个好天气,想出去玩，怎么办")
    code, ret=model.request()
    code1,dic=model.responsePares(code,ret)
    print(code1)
    print(dic)
























































































































