from langchain.chains.llm import LLMChain
from langchain_community.llms import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage, ChatMessage,SystemMessage
import telnetlib as telnet
from tools_helper.EnumConfig import TypeEnum
from tools_helper.ImageLoad import ImageProcessor
from llm_helper.img2text import  ImageGenerateText
from langchain_community.chat_models import ChatOllama
from PIL import ImageFilter
from matplotlib import pyplot as plt
modelObj=ImageGenerateText("100.120.0.28",11434,1)
url,code = modelObj.create_url_model()
imageIntsance =ImageProcessor()
ret=imageIntsance.image_load_window()
base64Str=imageIntsance.convert_to_base64(ret)
imgtemp=[]
imgtemp.append(base64Str)
#print(base64Str)
plt.imshow(ret)
plt.show()
#定义系统提示词
syscontent="你的名字叫啾啾，你的任务是帮助人们理解图像内容,回答的输出格式为json格式,并使用中文进行对话.如果客户配置了工具，你可以调用对应的工具。当你不知道如何回答客户的内容时，请回复<不能正确回答问题>。"

chatcontent="""
你需要根据提供的图像来进行问题的回答，回答问题的主要关注部分，请给出你的思考过程
"text":{text}
"image_url”:{image_url}
"""
hummsssagew=HumanMessage(content=[{"image_url":base64Str,'type':'image_url'},{"text":"这张图描述的什么内容",'type': 'text'}])

message=[SystemMessage(syscontent),hummsssagew]
print(message)

#chatTeplet=ChatPromptTemplate.from_messages([sysmsg,chatcontent])

#model=modelObj.create_model(url,"myvmodel:latest",sysmsg)
modelInstance=ChatOllama()
modelInstance.base_url=url
modelInstance.model="llava34B:latest"
modelInstance.top_p=0.2
modelInstance.top_k=200
modelInstance.temperature=0.2
modelInstance.num_ctx=9192
ret=modelInstance.invoke(message)
print(ret.content)