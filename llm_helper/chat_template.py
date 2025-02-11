# -*- coding: utf-8 -*-
# @Time    : 2024/6/25 16:34
# @Author  : zhanghaoxiang
# @File    : chat_template.py
# @Software: PyCharm
from langchain_core.messages import HumanMessage, ChatMessage,SystemMessage
from langchain_community.chat_models import ChatOllama
import langchain_core.messages.base
class ChatModel:
    def __init__(self,chatModel:ChatOllama,top_K:int,top_p:float,temperature:float):
        '''
        chatModel:
        :param chatModel:
        :param top_K:
        :param top_p:
        :param temperature:
        '''
        self.model=chatModel
        self.top_K=top_K
        self.top_p=top_p
        self.num_ctx=919200
        self.temperature=temperature

    def message(self,sys_Str:str,user_str:str,images:str)->[SystemMessage,HumanMessage]:
        '''
         创建会话消息
        :param sys_Str: 系统提示词，可以为空
        :param user_str: 用户问题
        :param images:   base64的图像编码
        :return: 返回会话消息
        '''
        sysMes=SystemMessage(content=sys_Str)
        humMes=HumanMessage(content=[{"image_url":images,'type':'image_url'},{"text":user_str,'type': 'text'}])
        chatMes=[]
        chatMes.append(sysMes)
        chatMes.append(humMes)
        return chatMes
    def response(self,message:[SystemMessage,HumanMessage])->str:
        try:
            result = self.model.invoke(message).content
            return result
        except Exception as e:
            return e




