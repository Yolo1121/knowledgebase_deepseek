# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import logging
from llm_helper.embeddding_template import EmbedddingTemplate

class PDFSearcher:
    def __init__(self):
        """初始化搜索器"""
        self.embedder = None
        self.vector_store = None
        
    def connect(self) -> bool:
        """建立连接"""
        try:
            self.embedder = EmbedddingTemplate()
            self.vector_store = self.embedder.create_embeddings_connect()
            return self.vector_store is not None
        except Exception as e:
            print(f"连接失败: {str(e)}")
            return False

    def search(self, query: str) -> Optional[Dict]:
        """
        执行搜索
        :param query: 搜索查询
        :return: 搜索结果字典或None
        """
        try:
            if not self.vector_store:
                raise Exception("未建立连接")

            print(f"\n执行搜索: '{query}'")
            
            # 预处理查询
            enhanced_query, expanded_terms, query_complexity, query_entities = self.embedder.preprocess_query(query)
            
            # 使用增强查询进行向量检索
            best_doc, best_score = self.embedder.vector_quray_embeddings(self.vector_store, enhanced_query)
            
            if not best_doc:
                print("未找到相关结果")
                return None

            # 获取内容和元数据
            content = best_doc.page_content  # 这是完整的原始内容
            metadata = best_doc.metadata

            if content:
                # 生成摘要
                summary = self.embedder._generate_summary(content, query)
                

                self.embedder._print_search_results(
                    query=enhanced_query,
                    content=content,
                    metadata=metadata,
                    score=best_score
                )
                
                # 使用 DeepSeek 基于摘要生成优化答案
                enhanced_answer = self.embedder.generate_enhanced_answer(
                    query=query,
                    full_answer=content,  # 使用完整内容
                    summary=summary
                )
                
                # 打印 DeepSeek 的输出
                print("\n" + "="*50)
                print("DeepSeek AI 优化答案:")
                print("-"*50)
                print(enhanced_answer)
                print("="*50)
                
                # 分析实体关系
                entity_analysis = self.embedder.analyze_entities(query, content)
                
                return {
                    'content': content,
                    'summary': summary,
                    'metadata': metadata,
                    'score': best_score,
                    'enhanced_query': enhanced_query,
                    'expanded_terms': expanded_terms,
                    'query_complexity': query_complexity,
                    'query_entities': query_entities,
                    'entity_analysis': entity_analysis,
                    'deepseek_answer': enhanced_answer
                }
            return None

        except Exception as e:
            print(f"搜索过程出错: {str(e)}")
            return None

    def __del__(self):
        """确保断开连接"""
        try:
            if hasattr(self, 'embedder'):
                self.embedder._disconnect()
        except Exception as e:
            if "NoneType" not in str(e):
                print(f"断开连接时出错: {e}")

def main():
    searcher = None
    try:
        # 初始化搜索器
        searcher = PDFSearcher()
        
        # 建立连接
        if not searcher.connect():
            print("无法连接到向量数据库")
            exit(1)
            
        while True:
            # 获取用户输入
            query = input("\n请输入搜索问题 (输入 'q' 退出): ")
            
            # 检查是否退出
            if query.lower() == 'q':
                print("退出搜索...")
                break
            
            # 执行搜索
            if query.strip():
                searcher.search(query)
            else:
                print("请输入有效的搜索问题")
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        # 安全断开连接
        if searcher:
            del searcher

if __name__ == "__main__":
    main()