# -*- coding: utf-8 -*-
# @Time    : 2024/6/25 17:48
# @Author  : zhanghaoxiang
# @File    : generate_template.py
# @Software: PyCharm
from langchain_community.llms import ollama
class GenerateModel:
    def __init__(self,genModel:ollama.Ollama,top_K:int,top_p:float,temperature:float):
        '''
        chatModel:
        :param chatModel:
        :param top_K:
        :param top_p:
        :param temperature:
        '''
        self.model=genModel
        self.top_K=top_K
        self.top_p=top_p
        self.temperature=temperature

    def message(self,user_str:str,images:str)->str:
        '''
         创建会话消息
        :param user_str: 用户问题
        :param images:   base64的图像编码
        :return: 返回会话消息
        '''
        message=(user_str,images)
        return
    def response(self,message:())->str:
        '''
        :param message: 包含用户问题的和传入的元组
        :return:
        '''
        try:
            result = self.model.invoke(input=message[0],images=message[1])
            return result
        except Exception as e:
            return e
