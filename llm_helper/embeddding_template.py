# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 8:53
# @Author  : zhanghaoxiang
# @File    : embeddding_template.py
# @Software: PyCharm
from langchain import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, Collection, utility
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
import PyPDF2
import time
import jieba.analyse
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import logging
import traceback
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from .pdf_md import pdf2md
from langchain_openai import ChatOpenAI
from openai import RateLimitError  # 新的导入方式
# 或者
import openai
from openai._exceptions import RateLimitError  # 另一种导入方式
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

jieba.setLogLevel(logging.INFO)  # 或者用 logging.WARNING

class EmbedddingTemplate:
    def __init__(self, host="localhost", port=19530, tb_name="sjznNew", device=None):
        """
        初始化向量模板类
        :param host: Milvus主机地址
        :param port: Milvus端口
        :param tb_name: 集合名称
        :param device: 设备类型 ('cuda' 或 'cpu')
        """
        self.host = host
        self.port = port
        self.tb_name = tb_name
        
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 初始化 BGE 模型
        try:
            model_name = "BAAI/bge-large-zh-v1.5"
            model_kwargs = {'device': self.device}
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            print("BGE 模型加载成功")
            
            # 测试 embedding 是否正常工作
            test_embedding = self.embeddings.embed_query("测试文本")
            if test_embedding is not None and len(test_embedding) > 0:
                print("Embedding 功能测试成功")
            
        except Exception as e:
            print(f"BGE 模型加载失败: {e}")
            raise  # 这是关键功能，加载失败就抛出异常

        # 初始化 NER 模型
        try:
            print("初始化 NER 模型...")
            
            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {self.device}")
            
            # 使用中文 BERT 模型
            model_name = "bert-base-chinese"  # 改用基础中文模型
            
            # 加载模型和分词器
            print(f"加载模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=7,  # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
                id2label={
                    0: "O",
                    1: "B-PER", 2: "I-PER",
                    3: "B-ORG", 4: "I-ORG",
                    5: "B-LOC", 6: "I-LOC"
                },
                label2id={
                    "O": 0,
                    "B-PER": 1, "I-PER": 2,
                    "B-ORG": 3, "I-ORG": 4,
                    "B-LOC": 5, "I-LOC": 6
                }
            )
            
            # 将模型移到正确的设备上
            self.ner_model = self.ner_model.to(self.device)
            
            # 设置模型为评估模式
            self.ner_model.eval()
            
            print("NER 模型加载成功")
            self.use_ner = True
                
        except Exception as e:
            print(f"NER 模型加载失败，详细错误: {str(e)}")
            self.tokenizer = None
            self.ner_model = None
            self.use_ner = False

        # 初始化jieba
        import jieba
        import jieba.posseg as pseg
        # 可以在这里添加自定义词典
        # jieba.load_userdict("path/to/dict.txt")
        
        # 增加带权重的同义词字典
        self.synonyms = {
            "销售": [("营业额", 0.9), ("营收", 0.8), ("收入", 0.7)],
            "成本": [("支出", 0.9), ("费用", 0.8), ("开销", 0.7)],
            "利润": [("盈利", 0.9), ("收益", 0.8), ("毛利", 0.7)],
            "增长": [("上升", 0.9), ("提升", 0.8), ("增加", 0.7)],
            "下降": [("减少", 0.9), ("降低", 0.8), ("下跌", 0.7)]
        }
        
        # 设置分段参数
        self.CHUNK_SIZE = 256  # 增加分段大小
        self.OVERLAP_SIZE = 50  # 增加重叠大小
        self.MIN_CHUNK_SIZE = 200  # 最小分段大小
        
        # 相似度阈值
        self.SIMILARITY_THRESHOLD = 0.85  # 去重相似度阈值

    def create_embeddings_connect(self) -> Optional[Milvus]:
        """创建向量数据库连接"""
        try:
            # 确保 embedding 模型已正确初始化
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                raise ValueError("Embedding 模型未正确初始化")

            # 创建向量存储
            vector_store = Milvus(
                embedding_function=self.embeddings,  # 使用已初始化的 embedding 函数
                collection_name=self.tb_name,
                connection_args={
                    "host": self.host,
                    "port": self.port
                }
            )
            print("向量数据库连接成功")
            return vector_store
            
        except Exception as e:
            print(f"连接向量数据库失败: {e}")
            return None

    def _disconnect(self):
        """安全断开连接"""
        try:
            if hasattr(self, 'embeddings'):
                connections.disconnect(alias="default")
                print("已断开向量数据库连接")
        except Exception as e:
            print(f"断开连接时出错: {e}")

    def _print_search_results(self, query: str, content: str, metadata: Dict, score: float):
        """打印搜索结果"""
        print("\n" + "="*50)
        print("搜索结果分析:")
        
        # 保持原始内容不变
        original_content = content
        
        # 生成摘要（使用优化后的_generate_summary方法）
        summary = self._generate_summary(content, query)
        
        # 计算各项指标前的预处理
        # 1. 扩展查询关键词（使用同义词）
        expanded_query = query
        query_keywords = jieba.analyse.extract_tags(query)
        for keyword in query_keywords:
            if keyword in self.synonyms:
                for synonym, weight in self.synonyms[keyword]:
                    expanded_query += f" {synonym}"
        
        # 2. 计算质量分数（使用扩展后的查询）
        quality_scores = self._evaluate_answer_quality(expanded_query, original_content, summary)
        
        print("\n1. 相似度评估:")
        print(f"问题与摘要答案的语义相似度: {quality_scores['summary_similarity']:.2%}")
        print(f"问题与完整答案的语义相似度: {quality_scores['full_similarity']:.2%}")
        print(f"摘要与完整答案的语义相似度: {quality_scores['summary_full_similarity']:.2%}")
        print(f"关键信息覆盖率: {quality_scores['info_coverage']:.2%}")
        print(f"实体匹配度: {quality_scores['entity_relevance']:.2%}")
        print(f"综合评分: {quality_scores['overall_score']:.2%}")
        
        if quality_scores.get('details'):
            print("\n2. 详细分析:")
            details = quality_scores['details']
            print("问题关键词:", ", ".join(details['query_keywords']))
            print("答案关键词:", ", ".join(details['answer_keywords']))
            print(f"答案句子数: {details['sentence_count']}")
            if details.get('query_entities'):
                print("问题实体:", ", ".join(details['query_entities']))
                print("答案实体:", ", ".join(details['answer_entities']))
        
        print("\n3. 文档内容:")
        print("-" * 50)
        print("摘要答案:")
        print(summary)
        print("\n完整文档片段:")
        print(original_content)
        print("-" * 50)

    def _generate_summary(self, text: str, query: str = None) -> str:
        """生成更智能的摘要答案"""
        try:
            # 1. 基本检查
            if not text or not isinstance(text, str):
                return text
            
            # 2. 分句
            sentences = [s.strip() for s in re.split('[。！？]', text) if s.strip()]
            if not sentences:
                return text
            
            # 3. 提取查询关键词
            query_keywords = set()
            if query:
                query_keywords = set(jieba.analyse.extract_tags(
                    query,
                    topK=5,
                    allowPOS=('n', 'v', 'vn', 'nz')
                ))
                # 扩展同义词
                expanded_keywords = set(query_keywords)
                for keyword in query_keywords:
                    if keyword in self.synonyms:
                        for synonym, weight in self.synonyms[keyword]:
                            expanded_keywords.add(synonym)
                query_keywords = expanded_keywords
            
            # 4. 对每个句子评分
            sentence_scores = []
            for sentence in sentences:
                score = 0
                sentence_words = set(jieba.cut(sentence))
                
                # 关键词匹配得分
                matched_keywords = query_keywords & sentence_words
                score += len(matched_keywords) * 2
                
                # 同义词匹配得分
                for word in sentence_words:
                    if word in self.synonyms:
                        score += 0.5
                
                # 实体匹配得分
                try:
                    sentence_entities = self._extract_entities_with_ner(sentence)
                    if query:
                        query_entities = self._extract_entities_with_ner(query)
                        entity_score = self._calculate_entity_relevance(query_entities, sentence_entities)
                        score += entity_score * 2
                except Exception as e:
                    print(f"实体处理失败: {e}")
                
                # 句子长度惩罚（避免过长句子）
                length_penalty = min(1.0, 50 / len(sentence)) if len(sentence) > 50 else 1.0
                score *= length_penalty
                
                sentence_scores.append((sentence, score))
            
            # 5. 选择最佳句子
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            selected_sentences = []
            current_length = 0
            max_length = 100  # 减小最大长度限制
            
            for sentence, score in sentence_scores:
                if score > 0 and current_length + len(sentence) <= max_length:  # 只选择有分数的句子
                    selected_sentences.append(sentence)
                    current_length += len(sentence)
                    if len(selected_sentences) >= 2:  # 最多选择2个句子
                        break
            
            # 6. 生成摘要
            if selected_sentences:
                summary = '。'.join(selected_sentences) + '。'
                return summary
            else:
                # 如果没有找到合适的句子，返回最短的句子
                shortest_sentence = min(sentences, key=len)
                return shortest_sentence + '。'
            
        except Exception as e:
            print(f"生成摘要时出错: {e}")
            return text[:100] + "..."  # 出错时返回截断的文本

    def process_document(self, text: str) -> List[str]:
        """优化的文档处理方法"""
        try:
            # 配置参数
            CHUNK_SIZE = 256  # 更小的分段大小
            OVERLAP_SIZE = 50  # 段落重叠大小
            
            # 1. 提取实体信息
            entities = self.extract_entities(text)
            
            # 2. 按自然语义边界分段
            paragraphs = []
            current_text = ""
            last_overlap = ""
            
            # 使用更细致的分隔规则
            segments = re.split(r'([。！？\n]+)', text)
            
            for i in range(0, len(segments)-1, 2):
                segment = segments[i] + (segments[i+1] if i+1 < len(segments) else '')
                
                # 添加上一段的重叠部分
                if last_overlap:
                    current_text = last_overlap + current_text
                
                if len(current_text) + len(segment) <= CHUNK_SIZE:
                    current_text += segment
                else:
                    if current_text:
                        # 保存当前段落的最后部分作为重叠
                        last_overlap = current_text[-OVERLAP_SIZE:] if len(current_text) > OVERLAP_SIZE else current_text
                        paragraphs.append(current_text)
                        current_text = segment
                    else:
                        current_text = segment
            
            if current_text:
                paragraphs.append(current_text)
            
            # 3. 增强段落内容
            enhanced_paragraphs = []
            for para in paragraphs:
                # 提取段落实体
                para_entities = self.extract_entities(para)
                
                # 提取关键词
                keywords = jieba.analyse.extract_tags(
                    para,
                    topK=5,
                    allowPOS=('n', 'v', 'vn', 'nz', 'nt', 'nr')
                )
                
                # 构建增强内容
                entity_info = []
                for entity_type, items in para_entities.items():
                    entity_texts = [item['text'] for item in items]
                    entity_info.append(f"{entity_type}: {', '.join(entity_texts)}")
                
                # 添加元数据
                enhanced_para = (
                    f"{para}\n"
                    f"[关键词: {', '.join(keywords)}]\n"
                    f"[实体信息: {'; '.join(entity_info)}]"
                )
                enhanced_paragraphs.append(enhanced_para)
            
            return enhanced_paragraphs
            
        except Exception as e:
            print(f"文档处理出错: {e}")
            return [text]

    def vector_quray_embeddings(self, instance: Milvus, query: str, top_k: int = 5) -> Tuple[Optional[Document], float]:
        """
        向量检索方法
        :param instance: Milvus向量存储实例
        :param query: 查询文本
        :param top_k: 返回结果数量
        :return: (最佳匹配文档, 相似度分数)
        """
        try:
            # 检查输入
            if not instance or not query:
                print("无效的输入参数")
                return None, 0.0

            print(f"开始检索...")
            print(f"集合名称: {self.tb_name}")
            
            # 首先建立连接
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
                print("已建立数据库连接")
            except Exception as e:
                if "already connected" not in str(e):
                    print(f"连接失败: {str(e)}")
                    return None, 0.0
            
            # 检查集合是否存在
            if utility.has_collection(self.tb_name):
                collection = Collection(self.tb_name)
                print(f"集合实体数量: {collection.num_entities}")
                
                # 尝试加载集合（不检查是否已加载）
                try:
                    collection.load()
                    print("集合已加载到内存")
                except Exception as e:
                    if "already loaded" not in str(e):
                        print(f"加载集合时出错: {str(e)}")
            else:
                print(f"警告: 集合 {self.tb_name} 不存在")
                return None, 0.0

            # 执行相似度搜索
            print("执行相似度搜索...")
            docs_and_scores = instance.similarity_search_with_score(
                query=query,
                k=top_k
            )

            if not docs_and_scores:
                print("未找到匹配结果")
                return None, 0.0

            print(f"找到 {len(docs_and_scores)} 个匹配结果")
            
            # 获取最佳匹配结果
            best_doc, best_score = docs_and_scores[0]


            return best_doc, best_score

        except Exception as e:
            print(f"向量检索失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            return None, 0.0

    def preprocess_query(self, query: str) -> tuple:
        """优化的查询预处理"""
        try:
            # 1. 扩大关键词提取范围
            keywords = jieba.analyse.extract_tags(
                query,
                topK=5,  # 增加到5个关键词
                allowPOS=('n', 'v', 'vn', 'nz', 'nt', 'nr', 'ns')  # 扩大词性范围
            )
            
            # 2. 实体识别
            query_entities = self.extract_entities(query)
            entity_texts = []
            entity_weights = {
                'PER': 1.2,  # 人名权重
                'ORG': 1.1,  # 组织机构权重
                'LOC': 1.1,  # 地名权重
                'TIME': 1.0  # 时间权重
            }
            
            # 收集实体文本并赋予权重
            weighted_entities = []
            for entity_type, items in query_entities.items():
                weight = entity_weights.get(entity_type, 1.0) #不存在则返回默认值 1.0
                for item in items:
                    entity_texts.append(item['text'])
                    weighted_entities.append((item['text'], weight))
            
            # 3. 同义词扩展
            expanded_terms = []
            for keyword in keywords + entity_texts:
                expanded_terms.append((keyword, 1.0))  # 原始关键词权重为1
                # 添加同义词
                if keyword in self.synonyms:
                    for synonym, weight in self.synonyms[keyword]:
                        expanded_terms.append((synonym, weight * 0.8))  # 同义词权重略低
            
            # 4. 构建增强查询
            enhanced_query_parts = [query]  # 基础查询
            enhanced_query_parts.extend(term for term, _ in expanded_terms)
            enhanced_query_parts.extend(term for term, _ in weighted_entities)
            enhanced_query = " ".join(enhanced_query_parts)
            
            # 5. 计算查询复杂度
            query_complexity = (len(keywords) + len(entity_texts)) / 10
            
            return enhanced_query, expanded_terms + weighted_entities, query_complexity, query_entities
            
        except Exception as e:
            print(f"查询预处理出错: {e}")
            return query, [], 0.5, {}

    def _calculate_entity_matches(self, query_entities: Dict, content: str) -> float:
        """
        计算查询实体在内容中的匹配度
        """
        try:
            if not query_entities or not content:
                return 0.0
            
            total_score = 0.0
            total_entities = 0
            content_lower = content.lower()
            
            for entity_type, entities in query_entities.items():
                for entity in entities:
                    total_entities += 1
                    entity_text = entity['text'].lower()
                    
                    # 完全匹配
                    if entity_text in content_lower:
                        total_score += 1.0
                    else:
                        # 检查部分匹配
                        words = entity_text.split()
                        if len(words) > 1:
                            matched_words = sum(1 for word in words if word in content_lower)
                            if matched_words > 0:
                                total_score += matched_words / len(words) * 0.8
            
            return total_score / total_entities if total_entities > 0 else 0.0
            
        except Exception as e:
            print(f"计算实体匹配时出错: {e}")
            return 0.0

    def _evaluate_results_quality(self, results, query):
        """评估检索结果质量"""
        if not results:
            return 0.5
        
        try:
            # 计算前三个结果的平均相似度
            query_vector = self.embeddings.embed_query(query)
            similarities = []
            
            for doc, _ in results[:3]:
                doc_vector = self.embeddings.embed_query(doc.page_content)
                similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                similarities.append(similarity)
            
            return np.mean(similarities)
        except:
            return 0.5

    def _update_results(self, all_results, seen_contents, doc, content, score):
        """智能去重和更新结果"""
        # 检查语义相似度
        for existing_content in seen_contents:
            similarity = self._calculate_similarity(content, existing_content)
            if similarity > self.SIMILARITY_THRESHOLD:
                # 如果新的得分更好，更新结果
                if score < seen_contents[existing_content]:
                    seen_contents[existing_content] = score
                    # 更新all_results中的得分
                    for i, (existing_doc, existing_score) in enumerate(all_results):
                        if existing_doc.page_content.strip() == existing_content:
                            all_results[i] = (existing_doc, score)
                return
        
        # 如果没有相似内容，添加新结果
        seen_contents[content] = score
        all_results.append((doc, score))

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        try:
            vec1 = self.embeddings.embed_query(text1)
            vec2 = self.embeddings.embed_query(text2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0

    def create_embeddings_from_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """从文本创建嵌入"""
        try:
            # 1. 文档分段
            chunks = self.process_document(text)
            
            # 2. 创建文档对象
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata=metadata or {"chunk": i}
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"创建嵌入出错: {e}")
            return []

    def add_documents(self, documents: List[Document], instance: Milvus = None):
        """添加文档到向量存储"""
        try:
            if not instance:
                instance = self.create_embeddings_connect()
            
            if not instance:
                raise Exception("无法创建向量存储连接")
            
            # 使用优化后的文档处理方法
            processed_docs = []
            for doc in documents:
                chunks = self.create_embeddings_from_text(
                    doc.page_content,
                    doc.metadata
                )
                processed_docs.extend(chunks)
            
            # 添加到向量存储
            instance.add_documents(processed_docs)
            print(f"成功添加 {len(processed_docs)} 个文档片段")
            
            return True
            
        except Exception as e:
            print(f"添加文档出错: {e}")
            return False

    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """优化的实体提取方法"""
        try:
            if not self.use_ner:
                print("NER模型未初始化，使用jieba进行实体识别")
                return self._extract_entities_with_jieba(text)
            
            try:
                # 使用NER 模型提取实体
                return self._extract_entities_with_ner(text)
                
            except Exception as e:
                print(f"NER处理出错，切换到jieba: {e}")
                return self._extract_entities_with_jieba(text)
            
        except Exception as e:
            print(f"实体提取出错: {e}")
            return {}

    def _extract_entities_with_jieba(self, text: str) -> Dict[str, List[Dict]]:
        """使用jieba进行实体识别"""
        try:
            import jieba.posseg as pseg
            words = pseg.cut(text)
            
            entities = {
                'PER': [],  # 人名
                'ORG': [],  # 组织
                'LOC': [],  # 地点
                'TIME': [], # 时间
                'MISC': []  # 其他
            }
            
            for word, flag in words:
                if flag.startswith('nr'):  # 人名
                    entities['PER'].append({'text': word, 'confidence': 0.8})
                elif flag.startswith('nt'):  # 机构名
                    entities['ORG'].append({'text': word, 'confidence': 0.8})
                elif flag.startswith('ns'):  # 地名
                    entities['LOC'].append({'text': word, 'confidence': 0.8})
                elif flag.startswith('t'):   # 时间词
                    entities['TIME'].append({'text': word, 'confidence': 0.8})
                elif flag.startswith('n'):   # 其他名词
                    entities['MISC'].append({'text': word, 'confidence': 0.6})
            
            return {k: v for k, v in entities.items() if v}
            
        except Exception as e:
            print(f"jieba实体提取出错: {e}")
            return {}

    def _extract_entities_with_ner(self, text: str) -> Dict[str, List[Dict]]:
        """使用 NER 模型提取实体"""
        try:
            # 初始化实体字典
            entities = {
                'PER': [],  # 人名
                'ORG': [],  # 组织
                'LOC': [],  # 地点
                'TIME': [], # 时间
                'MISC': []  # 其他
            }
            
            # 对文本进行编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True
            )
            
            # 保存 offset_mapping 但不传给模型
            offset_mapping = inputs.pop('offset_mapping')
            
            # 将输入移到正确的设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取模型预测
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
            
            # 获取预测结果
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # 处理预测结果
            current_entity = None
            current_text = ""
            
            for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
                if idx == 0 or idx == len(tokens) - 1:  # 跳过特殊标记
                    continue
                    
                pred_tag = self.ner_model.config.id2label[pred_id]
                
                if pred_tag.startswith("B-"):  # 新实体开始
                    if current_entity:  # 保存之前的实体
                        entity_type = current_entity
                        if entity_type in entities:
                            entities[entity_type].append({
                                'text': current_text.strip(),
                                'confidence': 0.8
                            })
                    
                    current_entity = pred_tag[2:]  # 去掉 "B-" 前缀
                    current_text = token
                    
                elif pred_tag.startswith("I-"):  # 实体继续
                    if current_entity and current_entity == pred_tag[2:]:
                        current_text += token.replace('##', '')
                        
                else:  # "O" 标签或其他
                    if current_entity:  # 保存之前的实体
                        entity_type = current_entity
                        if entity_type in entities:
                            entities[entity_type].append({
                                'text': current_text.strip(),
                                'confidence': 0.8
                            })
                        current_entity = None
                        current_text = ""
            
            # 处理最后一个实体
            if current_entity and current_text:
                entity_type = current_entity
                if entity_type in entities:
                    entities[entity_type].append({
                        'text': current_text.strip(),
                        'confidence': 0.8
                    })
            
            return {k: v for k, v in entities.items() if v}
            
        except Exception as e:
            print(f"NER 处理出错: {e}")
            return {}

    def _calculate_entity_relevance(self, query_entities: Dict, answer_entities: Dict) -> float:
        """优化的实体相关度计算"""
        try:
            if not query_entities or not answer_entities:
                return 0.0
            
            # 实体类型权重
            type_weights = {
                'PER': 1.2,   # 人名权重更高
                'ORG': 1.1,   # 组织机构次之
                'LOC': 1.1,   # 地名次之
                'TIME': 0.9,  # 时间权重较低
                'NUM': 0.8,   # 数字权重最低
                'MISC': 0.7   # 其他实体权重最低
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for entity_type in query_entities.keys():
                if entity_type not in answer_entities:
                    continue
                
                weight = type_weights.get(entity_type, 1.0)
                query_texts = {item['text'].lower() for item in query_entities[entity_type]}
                answer_texts = {item['text'].lower() for item in answer_entities[entity_type]}
                
                # 计算部分匹配
                for q_text in query_texts:
                    best_match_score = 0.0
                    for a_text in answer_texts:
                        # 完全匹配
                        if q_text == a_text:
                            best_match_score = 1.0
                            break
                        # 部分匹配
                        elif q_text in a_text or a_text in q_text:
                            best_match_score = max(best_match_score, 0.8)
                        # 字符重叠
                        else:
                            overlap = len(set(q_text) & set(a_text))
                            if overlap > 0:
                                best_match_score = max(best_match_score, overlap / max(len(q_text), len(a_text)))
                
                total_score += best_match_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            print(f"计算实体相关度时出错: {e}")
            return 0.0

    def _extract_code_entities(self, text: str) -> dict:
        """提取代码相关实体"""
        entities = {
            "函数定义": set(),
            "类定义": set(),
            "变量定义": set(),
            "导入语句": set(),
            "API调用": set()
        }
        
        try:
            # 匹配函数定义
            func_pattern = r'def\s+([a-zA-Z_]\w*)\s*\('
            entities["函数定义"].update(re.findall(func_pattern, text))
            
            # 匹配类定义
            class_pattern = r'class\s+([a-zA-Z_]\w*)\s*[:\(]'
            entities["类定义"].update(re.findall(class_pattern, text))
            
            # 匹配导入语句
            import_pattern = r'(?:from\s+[\w.]+\s+)?import\s+([\w.]+)'
            entities["导入语句"].update(re.findall(import_pattern, text))
            
            # 匹配变量定义
            var_pattern = r'([a-zA-Z_]\w*)\s*=\s*'
            entities["变量定义"].update(re.findall(var_pattern, text))
            
            # 匹配API调用
            api_pattern = r'([a-zA-Z_]\w*)\s*\.\s*[a-zA-Z_]\w*\s*\('
            entities["API调用"].update(re.findall(api_pattern, text))
            
            # 过滤掉空集合
            return {k: list(v) for k, v in entities.items() if v}
        except Exception as e:
            print(f"代码实体提取失败: {e}")
            return {}

    def _evaluate_answer_quality(self, query: str, answer: str, summary: str) -> Dict:
        """多维度评估答案质量"""
        try:
            # 初始化结果字典
            result = {
                'overall_score': 0.3,
                'summary_similarity': 0.0,
                'full_similarity': 0.0,
                'summary_full_similarity': 0.0,
                'info_coverage': 0.0,
                'entity_relevance': 0.0,
                'details': {
                    'query_keywords': [],
                    'answer_keywords': [],
                    'sentence_count': 0,
                    'query_entities': [],
                    'answer_entities': []
                }
            }

            # 1. 提取并扩展查询关键词
            query_keywords = set(jieba.analyse.extract_tags(query))
            expanded_query_keywords = set(query_keywords)
            for keyword in query_keywords:
                if keyword in self.synonyms:
                    for synonym, weight in self.synonyms[keyword]:
                        expanded_query_keywords.add(synonym)

            # 2. 提取答案关键词
            answer_keywords = set(jieba.analyse.extract_tags(answer))
            summary_keywords = set(jieba.analyse.extract_tags(summary))

            # 3. 计算关键信息覆盖率（使用扩展后的关键词）
            matched_keywords = expanded_query_keywords & answer_keywords
            result['info_coverage'] = len(matched_keywords) / len(expanded_query_keywords) if expanded_query_keywords else 0.0

            # 4. 计算实体相关度
            query_entities = self._extract_entities_with_ner(query)
            answer_entities = self._extract_entities_with_ner(answer)
            result['entity_relevance'] = self._calculate_entity_relevance(query_entities, answer_entities)

            # 5. 计算语义相似度
            query_vector = self.embeddings.embed_query(query)
            answer_vector = self.embeddings.embed_query(answer)
            summary_vector = self.embeddings.embed_query(summary)

            def cosine_similarity(v1, v2):
                return float(dot(v1, v2) / (norm(v1) * norm(v2)))

            result['summary_similarity'] = cosine_similarity(query_vector, summary_vector)
            result['full_similarity'] = cosine_similarity(query_vector, answer_vector)
            result['summary_full_similarity'] = cosine_similarity(summary_vector, answer_vector)

            # 6. 更新详细信息
            result['details']['query_keywords'] = list(expanded_query_keywords)
            result['details']['answer_keywords'] = list(answer_keywords)
            result['details']['sentence_count'] = len([s for s in re.split('[。！？]', answer) if s.strip()])
            result['details']['query_entities'] = query_entities
            result['details']['answer_entities'] = answer_entities

            # 7. 计算综合评分
            weights = {
                'summary_similarity': 0.3,
                'info_coverage': 0.3,
                'entity_relevance': 0.2,
                'completeness': 0.2
            }

            result['overall_score'] = (
                result['summary_similarity'] * weights['summary_similarity'] +
                result['info_coverage'] * weights['info_coverage'] +
                result['entity_relevance'] * weights['entity_relevance'] +
                min(1.0, result['details']['sentence_count'] / 3) * weights['completeness']
            )

            return result

        except Exception as e:
            print(f"评估答案质量时出错: {e}")
            return result

    def analyze_entities(self, query: str, content: str) -> Dict:
        """
        分析查询和内容中的实体关系
        """
        try:
            # 1. 提取实体
            query_entities = self.extract_entities(query)
            content_entities = self.extract_entities(content)
            
            # 2. 计算实体匹配度
            entity_matches = self._calculate_entity_matches(query_entities, content)
            
            # 3. 计算实体相关性
            entity_relevance = self._calculate_entity_relevance(query_entities, content_entities)
            
            # 4. 提取代码相关实体（如果有）
            code_entities = self._extract_code_entities(content)
            
            return {
                'entity_matches': entity_matches,
                'entity_relevance': entity_relevance,
                'query_entities': query_entities,
                'content_entities': content_entities,
                'code_entities': code_entities,
                'details': {
                    'matched_types': self._get_matched_entity_types(query_entities, content_entities),
                    'entity_coverage': self._calculate_entity_coverage(query_entities, content_entities)
                }
            }
            
        except Exception as e:
            logging.error(f"实体分析失败: {str(e)}")
            return {
                'entity_matches': 0.0,
                'entity_relevance': 0.0,
                'query_entities': {},
                'content_entities': {},
                'code_entities': {},
                'details': {
                    'matched_types': [],
                    'entity_coverage': 0.0
                }
            }

    def generate_enhanced_answer(self, query: str, full_answer: str, summary: str) -> str:
        """
        使用 DeepSeek 生成增强的答案
        """
        try:
            # DeepSeek API 配置
            api_url = "http://localhost:11434/api/generate"
            
            # 构建更精炼的提示模板
            prompt = f"""基于以下信息生成一个简洁的中文回答：

问题：{query}
相关文本：{full_answer}

要求：
1. 直接提取文本中的关键信息回答问题
2. 使用肯定的语气
3. 突出重点，去除冗余
4. 不要有任何解释和过渡语句
5. 控制在50字以内
6. 不要重复原文，要提炼核心信息"""
            
            # 发送请求到 DeepSeek API
            payload = {
                "model": "deepseek-r1:8b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # 降低温度，使输出更加确定
                "top_p": 0.1  # 降低随机性
            }
            
            try:
                response = requests.post(api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'response' in result and isinstance(result['response'], str):
                        answer = result['response'].strip()
                        
                        # 移除 <think> 等标记和多余的空行
                        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                        answer = re.sub(r'\n+', '\n', answer).strip()
                        
                        # 如果包含否定表述，尝试重新生成
                        if any(phrase in answer for phrase in ['未提及', '不清楚', '信息不足', '缺乏']):
                            # 使用更强调提取的提示重试
                            retry_prompt = f"""从以下文本中提取与问题相关的具体事实：

问题：{query}
文本：{full_answer}

要求：
1. 只提取已有的具体信息
2. 不要添加任何推测
3. 用最简洁的语言描述
4. 限制在30字以内"""

                            payload["prompt"] = retry_prompt
                            retry_response = requests.post(api_url, json=payload, timeout=60)
                            
                            if retry_response.status_code == 200:
                                retry_result = retry_response.json()
                                if 'response' in retry_result:
                                    answer = retry_result['response'].strip()
                                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                                    answer = re.sub(r'\n+', '\n', answer).strip()
                        
                        if answer and not any(phrase in answer for phrase in ['未提及', '不清楚', '信息不足', '缺乏']):
                            return answer
                
                # 如果生成失败，返回处理过的摘要
                return self._process_summary(summary)
                
            except requests.exceptions.RequestException as e:
                print(f"DeepSeek API 请求失败: {e}")
                return self._process_summary(summary)
            
        except Exception as e:
            print(f"生成增强答案时出错: {e}")
            return self._process_summary(summary)

    def _process_summary(self, summary: str) -> str:
        """处理摘要，提取核心信息"""
        try:
            # 分句
            sentences = [s.strip() for s in re.split('[。！？]', summary) if s.strip()]
            if not sentences:
                return summary
            
            # 提取关键词
            keywords = jieba.analyse.extract_tags(summary, topK=5)
            
            # 选择包含最多关键词的句子
            best_sentence = max(sentences, 
                              key=lambda s: sum(1 for k in keywords if k in s))
            
            return best_sentence
            
        except Exception as e:
            print(f"处理摘要时出错: {e}")
            return summary[:100]  # 返回截断的摘要

    def _get_matched_entity_types(self, query_entities: Dict, content_entities: Dict) -> List[str]:
        """
        获取查询和内容中匹配的实体类型
        """
        try:
            matched_types = []
            if not query_entities or not content_entities:
                return matched_types
            
            for entity_type in query_entities:
                if entity_type in content_entities:
                    query_texts = {item['text'].lower() for item in query_entities[entity_type]}
                    content_texts = {item['text'].lower() for item in content_entities[entity_type]}
                    
                    if query_texts & content_texts:  # 集合交集
                        matched_types.append(entity_type)
                    else:
                        # 检查部分匹配
                        for q_text in query_texts:
                            if any(q_text in c_text or c_text in q_text for c_text in content_texts):
                                matched_types.append(entity_type)
                                break
            
            return matched_types
            
        except Exception as e:
            print(f"获取匹配实体类型时出错: {e}")
            return []

    def _calculate_entity_coverage(self, query_entities: Dict, content_entities: Dict) -> float:
        """
        计算实体覆盖率
        """
        try:
            if not query_entities or not content_entities:
                return 0.0
            
            total_query_entities = sum(len(entities) for entities in query_entities.values())
            if total_query_entities == 0:
                return 0.0
            
            matched_count = 0
            for entity_type, query_items in query_entities.items():
                if entity_type not in content_entities:
                    continue
                
                content_texts = {item['text'].lower() for item in content_entities[entity_type]}
                for query_item in query_items:
                    query_text = query_item['text'].lower()
                    if query_text in content_texts or any(query_text in text or text in query_text 
                                                        for text in content_texts):
                        matched_count += 1
            
            return matched_count / total_query_entities
            
        except Exception as e:
            print(f"计算实体覆盖率时出错: {e}")
            return 0.0


# 只保留本地连接的测试
if __name__ == "__main__":
    # 只使用本地连接测试
    a = EmbedddingTemplate()  # 使用默认参数，确保是 localhost
    instance = a.create_embeddings_connect()
    