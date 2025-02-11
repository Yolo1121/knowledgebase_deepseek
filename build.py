import requests, json
#from tools_helper.PDF_Layout_Analysis import PDF_Layout_Analysis
from llm_helper.embeddding_template import EmbedddingTemplate
from buildtask import BuildTask,VectorTableBuildTask
from multiscalebuildtask import MultiscaleVectorTableBuildTask
from ragdoc import RagDoc

docinfourl = "http://100.120.0.27:9431"
def get_doc_info(docid):
    '''
    {
        "msg": "操作成功",
        "code": 200,
        "data": [
            {
                "createBy": null,
                "createTime": "2024-05-24 17:28:07",
                "updateBy": null,
                "updateTime": null,
                "remark": null,
                "id": 89,
                "userId": 2,
                "sessionId": 1,
                "roundId": 0,
                "question": "每个国家的歌手有多少",
                "answer": "{\"SQLQuery\":\"SELECT Country, COUNT(Singer_ID) AS 歌手数量 FROM singer GROUP BY Country LIMIT 100;\",\"answer\":[[\"France\",\"Netherlands\",\"United States\"],[4,1,1]],\"thinking\":\"\",\"columnnames\":[\"Country\",\"歌手数量\"],\"ploters\":[{\"type\":\"pie\",\"entityname\":\"Country\",\"entityvalue\":[\"France\",\"Netherlands\",\"United States\"],\"values\":[{\"name\":\"歌手数量\",\"data\":[4,1,1]}]},{\"type\":\"bar\",\"entityname\":\"Country\",\"entityvalue\":[\"France\",\"Netherlands\",\"United States\"],\"values\":[{\"name\":\"歌手数量\",\"data\":[4,1,1]}]}]}"
            }
        ]
    }
    '''
    param = {}
    param['docId'] = docid
    exceptnum = 0
    theException = None
    response = requests.get(
        docinfourl,  # type: ignore[arg-type]
        params=param,
    )
    if response.status_code == 200:
        t = json.loads(response.text)
        return t['data']
    else:
        response.raise_for_status()


def buildfunc(docjsonstr, taskjsonstr):
    docjson = json.loads(docjsonstr)
    taskjson = json.loads(taskjsonstr)
    vbt = None
    if "scalelist" in taskjson:
        vbt = MultiscaleVectorTableBuildTask(taskjson)
    else:
        vbt = VectorTableBuildTask(taskjson)
    if vbt is None:
        return "ERROR: buildtask cannot be empty!!!"
    print("start extact text !!!")
    ragdoc = RagDoc(docjson)
    textlist = ragdoc.extractext()
    print("finish extact text !!!")
    seginfo = vbt.synbuildtext(textlist)
    print("finish vector build !!!")
    vbt.savetextseginfo(seginfo)
    print("finish save sginfos !!!")
    return "succ"


if __name__ == "__main__":
    '''
    a=PDF_Layout_Analysis("/mnt/d/mycodes/tmp/zhongqi1.pdf","/mnt/d/mycodes/tmp/")
    ret,str1=a.load_pdf()
    li,code=a.convert_to_img(ret)
    a.image_parse(li)
    a.tables_2_images()
    test,b,doc=a.img_text_matching()
    print(test)
    print(b)
    print(doc)
    '''
    #开始进行向量入库
    a=EmbedddingTemplate(host="100.120.0.26",port=19530,tb_name="sjznNew")
    instance=a.create_embeddings_connect()
    textindex=a.vector_store_embeddings(instance,b)
    imagesindex=a.vector_store_embeddings(instance,doc)
    print(textindex)
    print(imagesindex)
