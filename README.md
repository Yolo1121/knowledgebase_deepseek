# PDF 文本解析与向量检索系统

本项目提供了一套完整的 PDF 文本解析、存储及检索方案，支持从 PDF 文档中提取文本数据并存入 Milvus 向量数据库，同时基于 `deepseek` 轻量模型 (`r1-8b`) 进行 AI 驱动的智能问答检索。

---

## 🚀 功能简介

- **PDF 解析**：使用 `tools_helper/PDF_Layout_Analysis.py` 提取 PDF 文本内容。
- **向量存储**：采用 **Milvus** 向量数据库存储解析后的文本数据。
- **数据验证**：使用 `test.py` 检查数据是否成功存入 Milvus，并返回存储的数据内容。
- **AI 检索**：使用pdf_search_vectortrans1.py 基于 `deepseek` (`r1-8b` 轻量模型) 进行文本检索及 AI 总结问答。
- **主要文件**：本项目涉及以下三个核心文件：
	1.	PDF_Layout_Analysis.py - 负责 PDF 文本解析和数据存储
	2.	pdf_search_vectortrans1.py - 负责向量检索及 AI 生成回答
	3.	embedding_template.py - 处理文本嵌入并转换为向量格式
  
          核心功能模块：EmbeddingTemplate 采用 BGE（BAAI/bge-large-zh-v1.5） 进行文本向量化，支持 批量文本处理、智能分段、自动去重，确保文本数据在高维空间中具备精准的语义表达。系统集成 BERT 和 jieba 分词，可进行 命名实体识别（NER），精准提取文本中的 人名、地名、机构名，同时支持 代码实体解析，适用于技术文档和代码分析场景。此外，EmbeddingTemplate 采用 Milvus 向量数据库，提供 高效的索引、检索、存储 方案，支持 数据持久化、增量更新与批量操作，确保数据管理的灵活性与稳定性。系统内置 语义相似度搜索，结合 多维度相似度计算、查询优化与缓存机制，确保检索结果精准高效，并能基于 深度学习模型 进行语义理解优化，提升搜索体验。

---

## 🛠 环境安装

### 1️⃣ 安装依赖

在项目根目录下运行以下命令安装所需的 Python 依赖包：

pip install -r requirements.txt

2️⃣ 部署 Milvus 向量数据库

本项目使用 Milvus 作为向量数据库，建议使用 Docker 进行本地部署。
	1.	拉取 Milvus 镜像（如果未安装 Docker，请先安装）：
```
docker pull milvusdb/milvus:latest
```
	2.	运行 Milvus 容器：
```
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -p 8080:8080 \
  -v /path/to/milvus/data:/var/lib/milvus \
  milvusdb/milvus:latest

注意：请替换 /path/to/milvus/data 为你希望存储 Milvus 数据的本地目录路径。
```
📌 使用指南

🔖 1 解析 PDF 并存入 Milvus

运行以下命令，解析 PDF 并将文本数据存入 Milvus 向量数据库：
```
python tools_helper/PDF_Layout_Analysis.py
```
📌 说明：
	•	该脚本会解析指定的 PDF 文档，提取文本信息，并向量化存入 Milvus 数据库。
	•	需要确保 Milvus 已成功启动。

🔖 2 验证数据是否成功存入 Milvus

执行以下命令检查数据是否成功存储，并查看返回的数据内容：
```
python test.py
```
🔖 3 下载 DeepSeek 轻量模型

本项目使用 deepseek r1-8b 轻量模型进行 AI 检索，请按照以下步骤下载模型：
	1.	访问 DeepSeek 官方网站：DeepSeek AI
	2.	下载 r1-8b 版本模型 并存放到本地目录
	3.	在代码中指定模型路径，确保程序能够正确加载

🔖 4 启动 AI 检索功能

运行以下命令启动 AI 向量检索服务：
```
python pdf_search_vectortrans1.py
```
📌 说明：
	•	运行后，按照提示输入查询问题，即可获得 AI 生成的答案和总结。
	•	确保 Milvus 已启动，并且 PDF 数据已经成功存入数据库。

📂 目录结构
```bash
ragbuild/
├── build.py
├── buildtask.py
├── clear_collection.py
├── deepseek_test.py
├── milvus-standalone-docker-compose.yml
├── multiscalebuildtask.py
├── myMilvus.py
├── pdf_search_vectortransform.py
├── ragdoc.py
├── README.md
├── requirements.txt
├── synserver.py
├── test_insert.py
├── test.py
├── tmp.pdf
├── vectorbuilder.py
├── __pycache__/
├── db/
├── demodocs/
├── llm_helper/
│   ├── __pycache__/
│   ├── output/
│   ├── __init__.py
│   ├── chat_template.py
│   ├── Convert_Pcd_bin.py
│   ├── embeddding_template.py
│   ├── embeddding_Text.py
│   ├── generate_template.py
│   ├── image_recover.py
│   ├── imageFore.py
│   ├── img2text.py
│   ├── load_model.py
│   ├── pcd_bin.py
│   ├── pdf_md.py
│   ├── table_rec.py
│   ├── TestExample.py
│   ├── TestExample2.py
│   └── text_ocr.py
└── tools_helper/
    ├── __pycache__/
    ├── output/
    ├── __init__.py
    ├── EnumConfig.py
    ├── ImageLoad.py
    ├── PDF_Layout_Analysis.py
    ├── tableImg2txt.py
    └── volumes/
```
最终使用结果展示：
<img width="812" alt="image" src="https://github.com/user-attachments/assets/953fea5d-bb3d-43bf-94ba-7d2f87ab16e6" />

如果你在使用过程中遇到问题，欢迎提交 Issue 进行反馈！🚀
