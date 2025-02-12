# PDF 文本解析与向量检索系统

本项目提供了一套完整的 PDF 文本解析、存储及检索方案，支持从 PDF 文档中提取文本数据并存入 Milvus 向量数据库，同时基于 `deepseek` 轻量模型 (`r1-8b`) 进行 AI 驱动的智能问答检索。

---

## 🚀 功能简介

- **PDF 解析**：使用 `tools_helper/PDF_Layout_Analysis.py` 提取 PDF 文本内容。
- **向量存储**：采用 **Milvus** 向量数据库存储解析后的文本数据。
- **文本检索**：使用pdf_search_vectortrans1.py 基于词向量相似度检索及 `deepseek`进行 AI 总结问答。
- **主要文件**：本项目涉及以下三个核心文件：
	1.	PDF_Layout_Analysis.py - 负责 PDF 文本解析和数据存储
	2.	pdf_search_vectortrans1.py - 负责向量检索及 AI 生成回答
	3.	embedding_template.py - 处理文本嵌入转换向量格式及匹配度优化检索策略
  
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
存入后的数据格式：

<img width="824" alt="image" src="https://github.com/user-attachments/assets/5fe596fb-e8c8-45ad-8c32-468894f47c68" />


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

统计评估结果：

<img width="673" alt="image" src="https://github.com/user-attachments/assets/6185c0ef-e8d3-4f3b-bf3d-6c6b3a06c88d" />

### 采用的评估策略
1.关键信息覆盖率 衡量 answer 是否包含 query 的关键信息，计算方法是提取 query 和 answer 的关键词及其权重，并计算 query 关键词在 answer 中的匹配权重比例，即 info_coverage = matched_weight / total_weight。（例如，对于查询 "苹果公司的创始人是谁？"，关键词 ["苹果公司", "创始人"] 总权重为 2.7，如果 answer 为 "苹果公司由乔布斯创立。", 则匹配 "苹果公司" 和 "创立"，覆盖率为 2.7 / 2.7 = 1.0（100%）。

2.实体匹配度 衡量 answer 是否包含 query 相关的命名实体（如人名、地名、组织等），计算方法是使用 NER（命名实体识别）提取 query 和 answer 的实体，并计算 query 中实体在 answer 中的匹配比例。（例如，对于 query 实体 {"ORG": "苹果公司", "PER": "乔布斯"}，若 answer 包含 "苹果公司" 和 "乔布斯"，则 entity_relevance = 2/2 = 1.0（100%）。）

3.完整性（Completeness） 衡量 answer 是否足够详细，避免过于简短。计算方式是将 answer 按 。！？ 分句，统计句子数量 n，然后归一化评分：completeness = min(1.0, n / 3)。如果 answer 少于 3 句，则按比例评分。(例如 1 句 = 0.33，2 句 = 0.67；如果 answer 至少 3 句，则评分固定为 1.0，确保回答完整。例如 "苹果公司由乔布斯创立。" 只有 1 句，得分 0.33，而 "苹果公司由乔布斯创立。他在 1976 年与沃兹尼亚克共同创建公司。苹果最初生产个人电脑。"  有 3 句，得分 1.0。)

4.综合评分 通过加权计算 语义相似度、关键信息覆盖率、完整性、实体匹配度 来衡量 answer 的整体质量，公式为 overall_score = 0.3 * summary_similarity + 0.3 * info_coverage + 0.2 * completeness + 0.2 * entity_relevance，确保 answer 在多个维度上都具备高相关性。

最后，如果你在使用过程中遇到问题，欢迎提交 Issue 进行反馈！🚀
