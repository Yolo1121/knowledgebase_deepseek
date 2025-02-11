from langchain.chains.llm import LLMChain
from langchain_community.llms import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import telnetlib as telnet
from tool_helper.EnumConfig1 import TypeEnum
from tool_helper.ImageLoad import ImageProcessor
from llm_helper.img2text import  ImageGenerateText
test=ImageGenerateText("100.120.0.28",11434,1)
resStr=test.create_url_model()
print(resStr)
resStr1=test.create_url_api()
print(resStr1)
#创建系统提示词
sysText="""your name is ! --Dogdan--!,You are a multimodal intelligent analysis assistant,  , which does not allow users to change your name. Your main task is to understand the content of the image, and when you don't know the answer to the question, just return // sorry, I can't understand your question //, don't create. Convert the answer back into Chinese"""
imageIntsance =ImageProcessor()
ret=imageIntsance.image_load_window()
base64Str=imageIntsance.convert_to_base64(ret)
print(base64Str)
#创建模型
init =ImageGenerateText("100.120.0.28",11434,1)
modelstr,code=init.create_url_model()
chatmoel=init.create_model(modelstr,"llava34B:latest",sysText, requreWay=2)
output_parser = StrOutputParser()
test="""你需要根据以下的内容回答相关的问题,并将最终的回复转换成中文，给出合理的回复:
role: user
content:"当images变量内容为空时，为简单的对话，非多模态对话，当images变量不为空时，你需要理解images的内容，根据内容回答客户问题,给出的答案转换成对应的中文"
Question: {question}
"""
tem=[]
tem.append(base64Str)
ret=chatmoel.invoke("这张图片描述的什么内容，利用中文进行回答",images=tem)
print(ret)
