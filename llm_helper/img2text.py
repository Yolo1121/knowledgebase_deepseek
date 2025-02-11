
from langchain_community.llms import ollama
import telnetlib as telnet
from langchain_community.chat_models import ChatOllama
from tools_helper.EnumConfig import TypeEnum
class ImageGenerateText(object):
    def __init__(self, ip:str, port:int, typeParam:int):
        self.ip = ip
        self.port = port
        self.type =typeParam
    def create_url_api(self)->(str,str):
        '''
        可以利用http的API形式去访问
        检测远端主机服务是否开启
        102 服务开启
        103 服务未开启
        :return: url/Erro， code
        '''
        try:
            instance = telnet.Telnet()
            instance.open(self.ip, self.port, timeout=6)
            print("远端服务程序端口已开启....")
            print("开始创建api服务")
            url_end=""
            for item in TypeEnum:

                if item.value==self.type:
                    url_end=item.name
            url="http://"+self.ip+":"+str(self.port)+"/api/"+url_end
            return (url, 102)
        except Exception as e:
            return (e,103)


    def create_model(self, url:str, modelname:str,systermPrompt:str,requreWay:int)->ollama.Ollama:
        #默认是生成模型
        vModel=ollama.Ollama()
        for item in TypeEnum:
            if item.value==requreWay and requreWay<3:
                if item.name=="chat":
                    vModel=ChatOllama()
        vModel.base_url=url
        vModel.model=modelname
        vModel.top_k=8192
        vModel.top_p=0.2
        vModel.temperature=0.3

        return vModel
    def create_url_model(self)->(str,str):
        '''
        可以利用远端模型形式去访问
        检测远端主机服务是否开启
        102 服务开启
        103 服务未开启
        :return: url/Erro， code
        '''
        try:
            instance = telnet.Telnet()
            instance.open(self.ip, self.port, timeout=6)
            print("远端服务程序端口已开启....")
            print("开始创建远端模型服务")
            url="http://"+self.ip+":"+str(self.port)
            return (url, 102)
        except Exception as e:
            return (e,103)














