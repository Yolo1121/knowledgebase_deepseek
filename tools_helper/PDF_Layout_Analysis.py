import os
import re
import json
import time
import cv2
import tarfile
import traceback  # 添加 traceback 导入
from rich.progress import track
import fitz
from PIL import Image
from paddleocr import PPStructure, save_structure_res, PaddleOCR
import sys
from pymilvus import connections, Collection, utility, DataType, CollectionSchema
from pymilvus import FieldSchema

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 修改导入语句，使用相对路径
from llm_helper.embeddding_template import EmbedddingTemplate
from llm_helper.chat_template import ChatModel
from llm_helper.img2text import ImageGenerateText
from llm_helper.text_ocr import TextOCR
from llm_helper.embeddding_Text import embedding_Text
from pymilvus import connections, Collection, utility
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tools_helper.ImageLoad import ImageProcessor
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def connect_milvus():
    """建立 Milvus 连接"""
    try:
        # 先断开可能存在的连接
        try:
            connections.disconnect("default")
        except:
            pass
            
        # 使用更简单的连接配置
        connections.connect(
            host="localhost",
            port=19530,
            timeout=30,
        )
        
        # 验证连接
        if connections.get_connection_addr("default"):
            print("Milvus 连接成功")
            return True
        return False
        
    except Exception as e:
        print(f"Milvus 连接错误: {e}")
        return False

class PDF_Layout_Analysis:
    '''
    :input_path: 文档输入路径
    :output_path:文档输出路径
    '''
    def __init__(self, input_path:str, output_path:str):
        # 在类初始化时添加警告过滤
        import warnings
        warnings.filterwarnings("ignore", message="Some weights of BertForTokenClassification")
        
        self.input_path = input_path
        self.output_path = output_path
        self.file_name = os.path.basename(self.input_path).split('.')[0]
        self.save_dir = os.path.join(self.output_path, f"{self.file_name}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 初始化向量存储相关属性
        self.embedder = None
        self.vector_store = None
        
    def init_vector_store(self):
        """初始化向量存储"""
        try:
            print("开始初始化向量存储...")
            
            # 初始化 embedding model
            self.embedder = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-large-zh",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 连接到 Milvus
            connections.connect(
                alias="default",
                host="localhost",
                port=19530
            )
            print("已连接到 Milvus")
            
            collection_name = "sjznNew"
            dim = 1024  # BGE-large-zh 模型的维度
            
            # 如果集合不存在才创建新的
            if not utility.has_collection(collection_name):
                # 创建集合 schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                
                schema = CollectionSchema(fields=fields, description="文档向量存储")
                collection = Collection(name=collection_name, schema=schema)
                
                # 创建索引
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                collection.create_index(field_name="vector", index_params=index_params)
                print(f"创建集合和索引成功: {collection_name}")
            else:
                collection = Collection(name=collection_name)
                print(f"使用现有集合: {collection_name}")
            
            # 加载集合
            collection.load()
            print("集合加载完成")
            
            # 初始化 Milvus 向量存储
            from langchain_community.vectorstores import Milvus
            self.vector_store = Milvus(
                embedding_function=self.embedder,
                collection_name=collection_name,
                connection_args={"host": "localhost", "port": 19530}
            )
            
            print("向量存储初始化完成")
            return True
            
        except Exception as e:
            print(f"初始化向量存储失败: {str(e)}")
            traceback.print_exc()
            return False

    def load_pdf(self):
        '''
        加载pdf文档
        :input_path:文档路径
        :return:
        '''
        try:
            pdf_document=fitz.open(self.input_path)
            return pdf_document,"300: File read successfully"
        except FileNotFoundError:
             return None,"301: File Not Found"
    def convert_to_img(self, pdf_document:fitz.Document):
        try:
            pageNums = pdf_document.page_count
            image_path_list = []
            for i in range(pageNums):
                page = pdf_document[i]
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                #识别图像的采用page_num_res文件夹存储
                dirNmae=f"{self.file_name}"+"_"+f"{i+1}_res"
                pageDir=os.path.join(self.save_dir,dirNmae)
                if not os.path.exists(pageDir):
                    os.makedirs(pageDir)
                image_path = os.path.join(pageDir, f"{self.file_name}_{i + 1}.jpg")
                image_path_list.append(image_path)
                image.save(image_path, "JPEG", quality=95)
                image.close()
                print("Process：当前完成转化页",i+1)
            return image_path_list, "302: page-->image:successfully"
        except Exception as e:
            return None, e
    def image_parse(self, image_path_list:[])->(int, str):
        img_path=[]
        img_path=image_path_list
        img_engine=PPStructure(table=False, ocr=False, show_log=True)
        for index, path in enumerate(img_path):
            print(f"当前正常进行第{index+1}页文档的版面分析，源文件地址：{path}")
            img=cv2.imread(path)
            result =img_engine(img)
            pagenum=index+1
            #存储到对应的不同文件夹下面
            dirname=f"{self.file_name}"+"_"+f"{index+1}_res"
            temdir=os.path.join(self.save_dir,dirname)
            save_structure_res(result,temdir,"parse_result", img_idx=pagenum)
            for line in result :
                line.pop('img')
                print(line)
        return 304, "304：版面结构分析完成"
    def table_parse(self,pdf_page_image,pdf_res_txt):
        dir_name = os.path.dirname(pdf_page_image)

        page_number = re.findall("\d+", pdf_page_image)[-1]
        with open(pdf_res_txt, 'r') as f:
            content = [json.loads(_.strip()) for _ in f.readlines()]

        figure_cnt = 1
        table_cnt = 1
        text_cnt = 1
        title_cnt = 1
        figurecaption_cnt = 1
        tablecaption_cnt = 1
        resdir="parse_result"
        for line in content:
            rect_type = line["type"]
            region = line["bbox"]
            # 将表格保存为图片
            if rect_type == "table":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{table_cnt}_table.jpg"
                    print(f"save table to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    table_cnt += 1
            if rect_type == "text":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{text_cnt}_text.jpg"
                    print(f"save text to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    text_cnt += 1


            if rect_type == "title":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{title_cnt}_title.jpg"
                    print(f"save title to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    title_cnt += 1
            '''
            if rect_type == "reference":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{table_cnt}_reference.jpg"
                    print(f"save reference to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    table_cnt += 1
            '''
            if rect_type == "figure_caption":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{figurecaption_cnt}_figurecaption.jpg"
                    print(f"save figure_caption to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    figurecaption_cnt += 1
            if rect_type == "table_caption":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{tablecaption_cnt}_tablecaption.jpg"
                    print(f"save figure_caption to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    tablecaption_cnt += 1
            if rect_type == "figure":
                with Image.open(pdf_page_image).convert('RGB') as image:
                    region_img = image.crop(region)
                    save_image_path = f"{dir_name}/{resdir}/{page_number}_{figure_cnt}_figure.jpg"
                    print(f"save table to {save_image_path}")
                    region_img.save(save_image_path, 'JPEG', quality=100)
                    figure_cnt += 1

    def tables_2_images(self):
        try:
            # 使用 os.path.join 来处理路径
            for file in os.listdir(self.save_dir):
                if os.path.isdir(os.path.join(self.save_dir, file)):
                    parse_result_dir = os.path.join(self.save_dir, file, "parse_result")
                    
                    # 确保目录存在
                    if not os.path.exists(parse_result_dir):
                        os.makedirs(parse_result_dir)
                        print(f"创建目录: {parse_result_dir}")
                        continue

                    print(f"处理目录: {parse_result_dir}")
                    for fileRet in os.listdir(parse_result_dir):
                        if(fileRet.split('.')[-1]=='txt'):
                            res_txt = fileRet.replace(self.file_name, "res").replace("jpg", "txt")

                            pdf_page_image_path = os.path.join(self.save_dir, file+f"/{self.file_name}_{file.split('_')[1]}.jpg")

                            pdf_res_txt_path = os.path.join(os.path.join(self.save_dir,file+"/parse_result"), res_txt)

                            self.table_parse(pdf_page_image=pdf_page_image_path,
                                         pdf_res_txt=pdf_res_txt_path)
                            print(f"当前处理第{file.split('_')[1]}个page")
                        else:
                            continue
        except Exception as e:
            print(f"处理表格时出错: {str(e)}")

    def img_text_matching(self):
        """文本提取和匹配"""
        try:
            print("开始文本提取...")
            textCollection = []
            # 初始化 PaddleOCR，优化参数
            ocr = PaddleOCR(
                use_angle_cls=True,      # 使用角度分类器
                lang="ch",               # 中文模型
                show_log=False,          # 不显示日志
                use_gpu=False,           # CPU模式
                det_db_thresh=0.1,       # 降低检测阈值，提高检测灵敏度
                det_db_box_thresh=0.1,   # 降低框检测阈值
                det_db_unclip_ratio=2.0, # 增加文本框扩张比例
                rec_char_dict_path=None, # 使用默认字典
                det_limit_side_len=4096, # 增加最大检测尺寸
                det_limit_type='max',    # 限制最长边
                rec_batch_num=6,         # 批处理数量
                drop_score=0.1,          # 降低文本置信度阈值
                min_subbox_size=10       # 最小文本框大小
            )
            
            for file in sorted(os.listdir(self.save_dir), 
                             key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                if not os.path.isdir(os.path.join(self.save_dir, file)):
                    continue
                    
                parse_result_dir = os.path.join(self.save_dir, file, "parse_result")
                if not os.path.exists(parse_result_dir):
                    continue
                
                page_text = ""
                page_num = file.split('_')[1]
                print(f"\n处理第 {page_num} 页...")
                
                text_files = sorted([f for f in os.listdir(parse_result_dir) if f.endswith('_text.jpg')],
                                  key=lambda x: int(x.split('_')[1]))
                
                for result_file in text_files:
                    path_str = os.path.join(parse_result_dir, result_file)
                    try:
                        # 读取图片
                        img = cv2.imread(path_str)
                        if img is None:
                            print(f"无法读取图片: {path_str}")
                            continue
                            
                        print(f"处理文件: {path_str}")
                        print(f"图片尺寸: {img.shape}")
                        
                        # 图像预处理
                        # 1. 转换为灰度图
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # 2. 二值化
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # 3. 降噪
                        denoised = cv2.fastNlMeansDenoising(binary)
                        # 4. 调整对比度
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray)
                        
                        # 对原图和增强后的图都进行OCR
                        for image in [img, enhanced]:
                            result = ocr.ocr(image, cls=True)
                            if result and len(result) > 0:
                                for idx in range(len(result)):
                                    if result[idx]:
                                        for line in result[idx]:
                                            if isinstance(line, list) and len(line) >= 2:
                                                text = line[1][0]
                                                confidence = line[1][1]
                                                print(f"识别文本: {text} (置信度: {confidence:.2f})")
                                                if confidence > 0.3:  # 降低置信度阈值
                                                    page_text += text.strip() + "\n"
                
                    except Exception as e:
                        print(f"处理文件 {path_str} 时出错: {str(e)}")
                        continue
                
                # 如果页面文本不为空，创建文档
                if page_text.strip():
                    cleaned_text = self._clean_text(page_text)
                    text_chunks = self._split_text_into_chunks(cleaned_text, chunk_size=512)
                    for i, chunk in enumerate(text_chunks):
                        if chunk.strip():
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "page_num": page_num,
                                    "chunk_num": i,
                                    "source": f"page_{page_num}_chunk_{i}",
                                    "type": "text"
                                }
                            )
                            textCollection.append(doc)
                    print(f"第 {page_num} 页处理完成，生成 {len(text_chunks)} 个文本块")
                else:
                    print(f"第 {page_num} 页未提取到有效文本")
            
            print(f"\n文本提取完成，共处理 {len(textCollection)} 条记录")
            return textCollection
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            traceback.print_exc()
            return []

    def _split_text_into_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """将长文本分割成指定大小的块"""
        if not text:
            return []
        
        # 按句子分割
        sentences = [s.strip() for s in re.split('[。！？]', text) if s.strip()]
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"
                
                # 如果单个句子超过chunk_size，强制分割
                while len(current_chunk) > chunk_size:
                    chunks.append(current_chunk[:chunk_size])
                    current_chunk = current_chunk[chunk_size-overlap:]
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def vector_store_embeddings(self, instance, documents):
        try:
            if not documents:
                print("警告: 没有文档需要存储")
                return None
            
            print(f"开始向量化和存储，文档数量: {len(documents)}")
            
            # 准备数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 生成向量
            embeddings = self.embedder.embed_documents(texts)
            
            # 生成唯一ID
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 准备插入数据
            entities = [
                ids,  # id 字段
                embeddings,  # vector 字段
                texts,  # text 字段
                metadatas  # metadata 字段
            ]
            
            # 获取集合
            collection = Collection("sjznNew")
            collection.insert(entities)
            
            print(f"存储成功，文档数量: {len(documents)}")
            return ids
            
        except Exception as e:
            print(f"存储向量时出错: {str(e)}")
            traceback.print_exc()
            return None

    def process_pdf(self):
        """完整的PDF处理流程"""
        try:
            print("开始处理PDF文件...")
            
            # 1. 加载PDF
            print("步骤1: 加载PDF")
            pdf_doc, status = self.load_pdf()
            if not pdf_doc:
                raise Exception(f"PDF加载失败: {status}")
            print("PDF加载成功")
            
            # 2. 转换为图片
            print("步骤2: 转换为图片")
            image_paths, status = self.convert_to_img(pdf_doc)
            if not image_paths:
                raise Exception(f"PDF转图片失败: {status}")
            print(f"转换成功，生成了 {len(image_paths)} 张图片")
            
            # 3. 版面分析
            print("步骤3: 版面分析")
            code, msg = self.image_parse(image_paths)
            if code != 304:
                raise Exception(f"版面分析失败: {msg}")
            print("版面分析完成")
            
            # 4. 提取区域图片
            print("步骤4: 提取区域图片")
            self.tables_2_images()
            print("区域图片提取完成")
            
            # 5. 文本提取和匹配
            print("步骤5: 文本提取和匹配")
            text_collection = self.img_text_matching()
            if not text_collection:
                raise Exception("未提取到文本内容")
            print(f"提取到 {len(text_collection)} 条文本记录")
            
            # 6. 初始化向量存储
            print("步骤6: 初始化向量存储")
            if not self.init_vector_store():
                raise Exception("向量存储初始化失败")
            print("向量存储初始化成功")
            
            # 7. 存储向量
            print("步骤7: 存储向量")
            text_index = self.vector_store_embeddings(self.vector_store, text_collection)
            if not text_index:
                raise Exception("向量存储失败")
            print("向量存储完成")
            
            return True, "处理完成"
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg
        finally:
            # 移除错误的清理代码
            try:
                connections.disconnect("default")
                print("已断开 Milvus 连接")
            except:
                pass

    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        try:
            # 删除多余的空白字符
            text = re.sub(r'\s+', ' ', text)
            # 删除特殊字符，但保留基本标点
            text = re.sub(r'[^\w\s。，！？：；""''（）、]', '', text)
            # 合并多个换行
            text = re.sub(r'\n+', '\n', text)
            return text.strip()
        except Exception as e:
            print(f"清理文本时出错: {str(e)}")
            return text

# 测试连接
if __name__ == "__main__":
    try:
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (ragbuild 的父目录)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # 设置输入输出路径
        pdf_file = os.path.join(project_root, "ragbuild/demodocs/mock文档/文档pdf/产品类/生鲜产品类文件（1）.pdf")
        output_dir = os.path.join(project_root, "output")
        
        print(f"当前目录: {current_dir}")
        print(f"项目根目录: {project_root}")
        print(f"PDF文件路径: {pdf_file}")
        print(f"输出目录: {output_dir}")
        
        # 检查文件是否存在
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_file}")
        
        # 创建处理实例
        analyzer = PDF_Layout_Analysis(pdf_file, output_dir)
        
        # 执行处理流程
        success, message = analyzer.process_pdf()
        print(f"\n处理结果: {message}")
        
    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()