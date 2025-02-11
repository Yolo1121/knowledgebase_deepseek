import sqlite3
from langchain import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

hfdict = {}
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hfdict["bge-large-zh"] = HuggingFaceBgeEmbeddings(
    model_name="/home/ssz/modelscope/hub/AI-ModelScope/bge-large-zh",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

class BuildTask:
    def __init__(self, taskjson, ragdoc):
        self.status = 0 # 0 空闲 1 构建中 2 构建完成
        self.doc = ragdoc
    def is_building(self):
        return self.status == 1
    def is_built(self):
        return self.status == 2
    def is_idle(self):
        return self.status == 0

class VectorTableBuildTask(BuildTask):
    def __init__(self, taskjson):
        super().__init__(taskjson, None)
        self.textsegwindowsize = taskjson["textsegwindowsize"]
        self.textsegslidestep = taskjson["textsegslidestep"]
        self.klgstorename = taskjson['klgstorename']
        self.klgtablename = taskjson['klgtablename']
        self.modelname = taskjson['modelname']
        self.embeddings = None
        self.vector_store = None
        self.host = taskjson['host']
        self.port = taskjson['port']
        try:
            self.embeddings = hfdict[self.modelname]
            self.vector_store = Milvus(embedding_function=self.embeddings,
                                       collection_name=self.klgstorename + "_" + self.klgtablename,
                                       connection_args={"host":self.host, "port":self.port},
                                       auto_id=True,
                                       metadata_field="info")
        except Exception as e:
            print(e)
    def synbuildtext(self, textlist, pkprefix="", textsegwindowsize=None, textsegslidestep=None):
        docs = [Document(page_content=textlist[i], metadata={"textlinenum": i}) for i in range(len(textlist))]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.textsegwindowsize if textsegwindowsize is None else textsegwindowsize,  # chunk size (characters)
            chunk_overlap=self.textsegslidestep if textsegslidestep is None else textsegslidestep,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        print("start vector store!!!")
        all_splits = text_splitter.split_documents(docs)
        print(len(all_splits))
        print(all_splits[-1].metadata["start_index"])
        pks = [pkprefix + "#textseg#%d#%d" % (doc.metadata["textlinenum"], doc.metadata["start_index"]) for doc in all_splits]
        vst = self.vector_store.add_documents(documents=all_splits, ids = pks)
        print("finish vector store!!!")
        return list(zip(pks, [x.page_content for x in all_splits]))
    def savetextseginfo(self, textseginfo):
        # 存入Sqlite
        conn = sqlite3.connect("db/docsegs.db")
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS '+self.klgstorename + "_" + self.klgtablename+' (pk text, content text)')
        cursor.executemany('INSERT INTO '+self.klgstorename + "_" + self.klgtablename+' VALUES (?,?)', textseginfo)
        conn.commit()   
        conn.close()