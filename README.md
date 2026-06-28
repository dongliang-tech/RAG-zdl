# 企业知识库 RAG 系统

## 目录

1. [项目概述](#1-项目概述)
2. [项目架构](#2-项目架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
5. [数据流与处理流程](#5-数据流与处理流程)
6. [配置说明](#6-配置说明)
7. [运行方式](#7-运行方式)
8. [依赖关系](#8-依赖关系)
9. [关键类与函数索引](#9-关键类与函数索引)

---

## 1. 项目概述

### 1.1 项目简介

本项目是一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）技术的企业知识库问答系统，源自 RAG Challenge 2 竞赛的获奖解决方案。系统能够对企业年报 PDF 文档进行智能解析、向量化存储，并基于检索到的上下文精准回答用户问题。

### 1.2 核心特性

- **多格式 PDF 解析**：支持 Docling 本地解析和 PDF MinerU 云端 API 解析
- **混合检索策略**：向量检索 + BM25 关键词检索 + LLM 重排 + Jina 重排
- **父文档检索**：支持 chunk 级检索和 page 级父文档返回
- **多模型支持**：OpenAI、Gemini、IBM Watson、阿里云 DashScope/Qwen
- **多问题类型**：支持字符串、数值、布尔、名单、比较类问题
- **表格智能处理**：表格序列化，将表格转换为独立信息块
- **可视化界面**：基于 Streamlit 的 Web 交互界面，支持明暗主题切换
- **并行处理**：支持多进程 PDF 解析和多线程问题处理
- **缓存加速**：Embedding 缓存 + Jina Rerank 缓存，显著提升响应速度

### 1.3 技术栈

| 类别 | 技术/框架 |
|------|-----------|
| 编程语言 | Python 3.11 |
| PDF 解析 | Docling、PDF MinerU |
| 向量数据库 | FAISS |
| 关键词检索 | BM25 (rank-bm25) |
| 重排服务 | Jina AI Reranker |
| LLM 接入 | OpenAI API、阿里云 DashScope、Gemini API |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| Web 框架 | Streamlit |
| CLI 框架 | Click |
| 数据验证 | Pydantic |

---

## 2. 项目架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                              │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  Streamlit UI   │  │         CLI 命令行                │  │
│  │ (app_streamlit) │  │           (main.py)              │  │
│  └────────┬────────┘  └──────────────┬───────────────────┘  │
└───────────┼──────────────────────────┼──────────────────────┘
            │                          │
┌───────────▼──────────────────────────▼──────────────────────┐
│                       业务编排层 (Pipeline)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Pipeline 类                        │   │
│  │  (流程控制 / 配置管理 / 阶段调度)                     │   │
│  └────────┬──────────┬──────────┬──────────┬───────────┘   │
└───────────┼──────────┼──────────┼──────────┼───────────────┘
            │          │          │          │
┌───────────▼──┐ ┌─────▼────┐ ┌──▼──────┐ ┌▼──────────────┐
│  文档处理层   │ │ 检索层   │ │ 重排层  │ │  答案生成层    │
│ (Ingestion)  │ │(Retrieval)│ │(Rerank)│ │ (Q.Processing) │
└──────┬───────┘ └─────┬─────┘ └────┬────┘ └───────┬───────┘
       │               │            │               │
┌──────▼───────┐ ┌─────▼─────┐ ┌────▼────┐ ┌──────▼───────┐
│  PDF 解析    │ │ 向量检索  │ │ LLM重排 │ │  LLM API     │
│  文本分块    │ │ BM25检索  │ │ Jina重排│ │  提示词工程  │
│  表格序列化  │ │ 混合检索  │ │         │ │  比较推理    │
└──────────────┘ └───────────┘ └─────────┘ └──────────────┘
```

### 2.2 核心数据流

```
PDF 文档
   │
   ▼
PDF 解析 (Docling / MinerU)
   │
   ▼
报告规整 + 表格序列化
   │
   ▼
文本分块 (Text Splitter)
   │
   ▼
向量化 + BM25 索引构建
   │
   ▼
向量数据库 (FAISS) + BM25 索引
   │
   ▼
用户问题
   │
   ▼
公司名提取 → 检索 (向量/BM25/混合) → 重排 (LLM)
   │
   ▼
RAG 上下文构建
   │
   ▼
LLM 答案生成 (结构化输出)
   │
   ▼
答案 + 引用来源
```

---

## 3. 目录结构

```
RAG-zdl/
├── data/
│   └── stock_data/                     # 股票数据示例
│       ├── pdf_reports/                # PDF 年报文件
│       ├── databases/
│       │   ├── chunked_reports/        # 分块后的报告 (JSON)
│       │   └── vector_dbs/             # FAISS 向量数据库
│       ├── debug_data/
│       │   └── 03_reports_markdown/    # 转换后的 Markdown
│       ├── questions.json              # 问题列表
│       ├── subset.csv                  # 公司元数据
│       └── answers_*.json              # 生成的答案
├── src/                                # 核心源代码
│   ├── pipeline.py                     # 主流程编排
│   ├── pdf_parsing.py                  # PDF 解析 (Docling)
│   ├── pdf_mineru.py                   # PDF 解析 (MinerU API)
│   ├── parsed_reports_merging.py       # 报告文本规整
│   ├── text_splitter.py                # 文本分块
│   ├── ingestion.py                    # 向量库/BM25 构建
│   ├── retrieval.py                    # 检索模块
│   ├── reranking.py                    # 重排模块
│   ├── questions_processing.py         # 问题处理
│   ├── api_requests.py                 # LLM API 封装
│   ├── api_request_parallel_processor.py # 异步 API 处理
│   ├── prompts.py                      # 提示词模板
│   └── tables_serialization.py         # 表格序列化
├── docs/
│   └── src_modules_overview.md         # 模块概览文档
├── app_streamlit.py                    # Streamlit Web UI
├── main.py                             # CLI 入口
├── requirements.txt                    # Python 依赖
├── env                                 # 环境变量模板
└── README.md                           # 项目说明
```

---

## 4. 核心模块详解

### 4.1 主流程编排 - pipeline.py

**文件位置**：[src/pipeline.py](src/pipeline.py)

#### 核心类

##### `PipelineConfig`
路径配置类，管理所有数据目录和文件路径。

**主要属性**：
- `root_path` - 数据根目录
- `pdf_reports_dir` - PDF 报告目录
- `questions_file_path` - 问题文件路径
- `answers_file_path` - 答案输出路径
- `vector_db_dir` - 向量数据库目录
- `documents_dir` - 分块文档目录
- `reports_markdown_path` - Markdown 报告目录

##### `RunConfig`
运行配置类，定义管道各阶段的开关和参数。

**主要属性**：
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_serialized_tables` | bool | False | 是否使用序列化表格 |
| `parent_document_retrieval` | bool | False | 是否启用父文档检索 |
| `use_vector_dbs` | bool | True | 是否使用向量数据库 |
| `use_bm25_db` | bool | False | 是否使用 BM25 检索 |
| `llm_reranking` | bool | False | 是否启用 LLM 重排 |
| `llm_reranking_sample_size` | int | 30 | 重排采样数量 |
| `top_n_retrieval` | int | 10 | 检索返回 top N |
| `parallel_requests` | int | 1 | 并行请求数 |
| `api_provider` | str | "dashscope" | API 提供商 |
| `answering_model` | str | "qwen-turbo-latest" | 回答模型 |

##### `Pipeline`
主流程类，编排整个 RAG 管道的各个阶段。

**核心方法**：

| 方法 | 说明 |
|------|------|
| `parse_pdf_reports()` | 解析 PDF 报告（支持并行） |
| `export_reports_to_markdown()` | 使用 MinerU 将 PDF 转 Markdown |
| `chunk_reports()` | 对报告进行文本分块 |
| `create_vector_dbs()` | 创建 FAISS 向量数据库 |
| `create_bm25_db()` | 创建 BM25 索引 |
| `process_parsed_reports()` | 完整报告处理流程（分块+建库） |
| `process_questions()` | 批量处理问题并生成答案 |
| `answer_single_question()` | 单条问题即时推理 |

**预设配置**：
- `base_config` - 基础配置（向量检索 + 路由）
- `pdr_config` - 父文档检索配置
- `max_config` - 最强配置（序列化表格 + 父文档检索 + LLM 重排）

---

### 4.2 PDF 解析模块

#### 4.2.1 pdf_parsing.py (Docling 本地解析)

**文件位置**：[src/pdf_parsing.py](src/pdf_parsing.py)

##### `PDFParser` 类
基于 Docling 库的 PDF 文档解析器。

**核心功能**：
- 调用 Docling 进行 PDF 结构解析
- 支持 OCR 文字识别
- 表格结构识别与提取
- 多进程并行解析

**核心方法**：
| 方法 | 说明 |
|------|------|
| `convert_documents()` | 批量转换 PDF 文档 |
| `process_documents()` | 处理转换结果并导出 JSON |
| `parse_and_export()` | 完整解析并导出流程 |
| `parse_and_export_parallel()` | 多进程并行解析 |

##### `JsonReportProcessor` 类
将 Docling 输出转换为标准化报告格式。

**核心功能**：
- 组装元信息（sha1、公司名）
- 按页组装内容（文本、表格、图片）
- 表格转换为 Markdown/HTML
- 调试数据导出

---

#### 4.2.2 pdf_mineru.py (MinerU 云端解析)

**文件位置**：[src/pdf_mineru.py](src/pdf_mineru.py)

**核心函数**：
| 函数 | 说明 |
|------|------|
| `get_task_id(file_name)` | 提交 PDF 解析任务，返回 task_id |
| `get_result(task_id)` | 轮询获取解析结果并下载 |
| `unzip_file(zip_path)` | 解压下载的结果文件 |

**特点**：
- 基于 MinerU 云端 API
- 支持 OCR 和公式识别
- 返回结构化 Markdown 格式

---

### 4.3 报告规整模块 - parsed_reports_merging.py

**文件位置**：[src/parsed_reports_merging.py](src/parsed_reports_merging.py)

##### `PageTextPreparation` 类
负责按规则清洗和格式化页面块。

**核心功能**：
- 过滤页脚、图片等不需要的块
- 文本清洗（OCR 错误修正、特殊字符替换）
- 表格组、列表组、脚注的连续合并
- Markdown 标题层级格式化
- 可选插入序列化表格

**核心方法**：
| 方法 | 说明 |
|------|------|
| `process_report()` | 处理单份报告 |
| `prepare_page_text()` | 单页文本组装主流程 |
| `_apply_formatting_rules()` | 应用格式化规则 |
| `_render_table_group()` | 渲染表格组 |
| `_render_list_group()` | 渲染列表组 |

---

### 4.4 文本分块模块 - text_splitter.py

**文件位置**：[src/text_splitter.py](src/text_splitter.py)

##### `TextSplitter` 类
文本分块工具类，支持多种分块策略。

**核心功能**：
- 基于 tiktoken 的 token 计数
- LangChain RecursiveCharacterTextSplitter 分块
- 按页分块 + 表格块插入
- Markdown 文件按行分块
- 元信息补充（company_name、sha1）

**核心方法**：
| 方法 | 说明 |
|------|------|
| `split_all_reports()` | 批量处理 JSON 报告 |
| `split_markdown_reports()` | 批量处理 Markdown 文件 |
| `split_markdown_file()` | 单个 Markdown 文件分块 |
| `_split_page()` | 单页文本分块 |
| `count_tokens()` | 统计 token 数 |

**分块参数**：
- 默认 chunk_size: 300 tokens
- 默认 chunk_overlap: 50 tokens
- Markdown 模式: 按行分割（默认 30 行/块，重叠 5 行）

---

### 4.5 索引构建模块 - ingestion.py

**文件位置**：[src/ingestion.py](src/ingestion.py)

#### 4.5.1 `VectorDBIngestor` 类
向量数据库构建工具。

**核心功能**：
- 调用 DashScope TextEmbedding API 获取嵌入向量
- 使用 FAISS 构建向量索引（内积/余弦相似度）
- 支持批量嵌入（每批 25 条）
- 自动重试机制（tenacity）

**核心方法**：
| 方法 | 说明 |
|------|------|
| `_get_embeddings()` | 获取文本嵌入向量 |
| `_create_vector_db()` | 创建 FAISS 索引 |
| `process_reports()` | 批量处理报告并保存向量库 |

**嵌入模型**：`text-embedding-v1` (DashScope)

---

#### 4.5.2 `BM25Ingestor` 类
BM25 关键词索引构建工具。

**核心功能**：
- 基于 rank-bm25 库构建 BM25Okapi 索引
- 按报告独立保存为 pickle 文件

**核心方法**：
| 方法 | 说明 |
|------|------|
| `create_bm25_index()` | 创建 BM25 索引 |
| `process_reports()` | 批量处理报告并保存索引 |

---

### 4.6 检索模块 - retrieval.py

**文件位置**：[src/retrieval.py](src/retrieval.py)

#### 4.6.1 `VectorRetriever` 类
向量检索器。

**核心功能**：
- 加载所有 FAISS 向量库和对应文档
- 支持 OpenAI 和 DashScope 两种嵌入提供商
- 按公司名定向检索
- 支持父文档（整页）返回模式

**核心方法**：
| 方法 | 说明 |
|------|------|
| `retrieve_by_company_name()` | 按公司名检索 top N 文本块 |
| `retrieve_all()` | 返回公司所有页面内容 |
| `_get_embedding()` | 获取查询文本的向量 |
| `_load_dbs()` | 加载所有向量数据库 |

---

#### 4.6.2 `BM25Retriever` 类
BM25 关键词检索器。

**核心功能**：
- 按公司名加载对应 BM25 索引
- 基于关键词相似度检索
- 支持父文档返回模式

**核心方法**：
| 方法 | 说明 |
|------|------|
| `retrieve_by_company_name()` | 按公司名 BM25 检索 |

---

#### 4.6.3 `HybridRetriever` 类
混合检索器（向量检索 + LLM 重排）。

**核心功能**：
- 先向量检索获取候选集
- 再使用 LLM 进行语义重排
- 加权融合向量分数和 LLM 相关性分数

**核心方法**：
| 方法 | 说明 |
|------|------|
| `retrieve_by_company_name()` | 混合检索 + 重排 |

---

### 4.7 重排模块 - reranking.py

**文件位置**：[src/reranking.py](src/reranking.py)

#### 4.7.1 `LLMReranker` 类
基于大模型的重排器。

**核心功能**：
- 单文档评分和批量文档评分两种模式
- 向量相似度与 LLM 相关性加权融合
- 多线程并行处理
- 支持 OpenAI 和 DashScope 两种提供商

**核心方法**：
| 方法 | 说明 |
|------|------|
| `get_rank_for_single_block()` | 单个文本块相关性评分 |
| `get_rank_for_multiple_blocks()` | 批量文本块相关性评分 |
| `rerank_documents()` | 文档重排主方法（加权融合） |

**评分机制**：
- 融合分数 = `llm_weight * relevance_score + (1 - llm_weight) * vector_distance`
- 默认 `llm_weight = 0.7`

---

#### 4.7.2 `JinaReranker` 类
基于 Jina API 的重排器（多语言支持）。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `rerank()` | 调用 Jina API 进行重排 |

---

### 4.8 问题处理模块 - questions_processing.py

**文件位置**：[src/questions_processing.py](src/questions_processing.py)

##### `QuestionsProcessor` 类
问题处理主类，编排从检索到答案生成的完整流程。

**核心功能**：
- 单公司问题处理
- 多公司比较问题处理（问题拆解 + 并行检索 + 汇总比较）
- 批量问题处理（支持并行和断点保存）
- 单条问题即时推理
- 页码引用校验与去幻觉
- 答案详情保存（分步推理、引用页等）

**核心方法**：
| 方法 | 说明 |
|------|------|
| `process_question()` | 处理单个问题（自动判断单/多公司） |
| `get_answer_for_company()` | 单个公司的答案生成 |
| `process_comparative_question()` | 多公司比较问题处理 |
| `process_all_questions()` | 批量处理所有问题 |
| `process_single_question()` | 单条问题即时推理 |
| `_validate_page_references()` | 校验引用页码，去幻觉 |
| `_extract_companies_from_subset()` | 从问题中提取公司名 |

**比较问题处理流程**：
1. LLM 将比较问题拆解为各公司独立问题
2. 并行处理每个公司的独立问题
3. 汇总各公司答案，LLM 生成比较结论

---

### 4.9 API 请求模块 - api_requests.py

**文件位置**：[src/api_requests.py](src/api_requests.py)

#### 4.9.1 多提供商支持

| 提供商 | 基础类 | 说明 |
|--------|--------|------|
| OpenAI | `BaseOpenaiProcessor` | GPT-4o / GPT-4o-mini / o3-mini |
| DashScope | `BaseDashscopeProcessor` | Qwen-Turbo / Qwen-Plus |
| IBM Watson | `BaseIBMAPIProcessor` | Llama 系列模型 |
| Gemini | `BaseGeminiProcessor` | Gemini 2.0 系列 |

#### 4.9.2 `APIProcessor` 类
统一 API 入口，根据 provider 路由到对应处理器。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `send_message()` | 发送消息到 LLM |
| `get_answer_from_rag_context()` | 基于 RAG 上下文生成答案 |
| `get_rephrased_questions()` | 拆解比较问题 |
| `_build_rag_context_prompts()` | 构建 RAG 提示词 |

#### 4.9.3 `AsyncOpenaiProcessor` 类
异步批量 OpenAI 请求处理器。

**核心功能**：
- 基于 JSONL 文件的异步请求处理
- 速率限制（每分钟请求数/token 数）
- 自动重试
- 进度监控

---

### 4.10 提示词模块 - prompts.py

**文件位置**：[src/prompts.py](src/prompts.py)

#### 提示词分类

| 提示词类 | 用途 | Schema 类型 |
|----------|------|-------------|
| `RephrasedQuestionsPrompt` | 比较问题拆解 | 重写问题列表 |
| `AnswerWithRAGContextStringPrompt` | 文本类答案 | string |
| `AnswerWithRAGContextNumberPrompt` | 数值类答案 | number |
| `AnswerWithRAGContextBooleanPrompt` | 布尔类答案 | boolean |
| `AnswerWithRAGContextNamePrompt` | 名称类答案 | name |
| `AnswerWithRAGContextNamesPrompt` | 名单类答案 | names |
| `ComparativeAnswerPrompt` | 比较类答案 | comparative |
| `RerankingPrompt` | 重排评分 | 相关性分数 |
| `AnswerSchemaFixPrompt` | JSON 修复 | 格式化助手 |

#### 统一答案结构

所有 RAG 答案都遵循以下结构：
```python
{
    "step_by_step_analysis": str,    # 分步推理过程（至少5步，150字以上）
    "reasoning_summary": str,         # 推理摘要（约50字）
    "relevant_pages": List[int],      # 相关页面编号
    "final_answer": Any               # 最终答案（类型取决于schema）
}
```

---

### 4.11 表格序列化模块 - tables_serialization.py

**文件位置**：[src/tables_serialization.py](src/tables_serialization.py)

##### `TableSerializer` 类
表格智能序列化工具。

**核心功能**：
- 将表格转换为上下文独立的信息块
- 提取表格上下文（前后文本）
- 支持同步和异步两种处理模式
- 多线程并行处理

**核心方法**：
| 方法 | 说明 |
|------|------|
| `serialize_tables()` | 同步序列化报告中所有表格 |
| `async_serialize_tables()` | 异步批量序列化 |
| `process_directory_parallel()` | 多线程处理目录 |
| `_get_table_context()` | 获取表格所在页上下文 |

**序列化输出结构**：
```python
{
    "subject_core_entities_list": List[str],  # 核心实体列表
    "relevant_headers_list": List[str],       # 相关表头列表
    "information_blocks": [                   # 信息块列表
        {
            "subject_core_entity": str,       # 核心主体
            "information_block": str          # 完整上下文的信息块
        }
    ]
}
```

---

### 4.12 Web UI 模块 - app_streamlit.py

**文件位置**：[app_streamlit.py](app_streamlit.py)

##### Streamlit 界面
基于 Streamlit 的交互式 Web 界面。

**界面组成**：
- **左侧边栏**：问题输入框 + 生成按钮
- **主内容区**：检索结果展示
  - 分步推理
  - 推理摘要
  - 相关页面
  - 最终答案

**使用方式**：
```bash
streamlit run app_streamlit.py
```

---

## 5. 数据流与处理流程

### 5.1 数据处理全流程

```
阶段1: PDF 解析
  ├─ 输入: PDF 文件
  ├─ 方式: Docling 本地 / MinerU 云端
  └─ 输出: 结构化 JSON / Markdown

阶段2: 报告规整
  ├─ 输入: 解析后 JSON
  ├─ 处理: 文本清洗 + 格式规整 + 表格序列化
  └─ 输出: 规整后报告 JSON

阶段3: 文本分块
  ├─ 输入: 规整后报告
  ├─ 处理: 按 token/行 分块 + 元信息补充
  └─ 输出: chunked_reports (JSON)

阶段4: 索引构建
  ├─ 输入: 分块报告
  ├─ 处理: 向量化 (FAISS) + BM25 索引
  └─ 输出: vector_dbs / bm25_dbs

阶段5: 问题处理
  ├─ 输入: 用户问题
  ├─ 处理: 
  │   ├─ 公司名提取
  │   ├─ 检索 (向量/BM25/混合)
  │   ├─ LLM 重排
  │   ├─ RAG 上下文构建
  │   └─ LLM 答案生成
  └─ 输出: 结构化答案 + 引用来源
```

### 5.2 问题处理详细流程

```
用户问题
    │
    ▼
[公司名提取]
    ├─ 从 subset.csv 读取公司列表
    └─ 在问题文本中匹配公司名
    │
    ▼
[数量判断] ── 1家 ──► [单公司处理]
    │                        │
    │                        ▼
    │                   [检索]
    │                   向量检索 → (可选) LLM重排
    │                        │
    │                        ▼
    │                   [上下文构建]
    │                        │
    │                        ▼
    │                   [LLM 答案生成]
    │                   (结构化输出 + CoT推理)
    │                        │
    │                        ▼
    │                   [引用校验]
    │                   (去幻觉 + 页码验证)
    │                        │
    │                        └──────┐
    │                               │
    └── 多家 ──► [比较问题处理] ◄───┘
                    │
                    ▼
              [问题拆解]
              (LLM 拆分为各公司独立问题)
                    │
                    ▼
              [并行处理]
              (每个公司走单公司流程)
                    │
                    ▼
              [汇总比较]
              (LLM 基于各公司答案生成结论)
                    │
                    ▼
              最终答案
```

---

## 6. 配置说明

### 6.1 环境变量配置

复制 `env` 文件为 `.env` 并填入相应密钥：

```env
OPENAI_API_KEY=sk-...          # OpenAI API 密钥
GEMINI_API_KEY=AIza...          # Gemini API 密钥
JINA_API_KEY=jina_...           # Jina Reranker API 密钥（用于文档重排）
DASHSCOPE_API_KEY=sk-...        # 阿里云 DashScope API 密钥（用于 Embedding 和 LLM）
```

### 6.2 缓存配置

系统支持以下缓存机制以提升响应速度：

| 缓存类型 | 缓存位置 | 说明 |
|----------|----------|------|
| Embedding 缓存 | `data/stock_data/databases/cache/embedding_cache.json` | 基于 query MD5 hash 缓存向量结果 |
| Jina Rerank 缓存 | `data/stock_data/databases/cache/jina_cache.json` | 基于 query+文档组合 hash 缓存重排结果 |

缓存会在相同查询时自动命中，无需额外配置。

### 6.3 运行配置 (RunConfig)

#### 预设配置对比

| 配置项 | base | pdr | max |
|--------|------|-----|-----|
| 序列化表格 | ❌ | ❌ | ✅ |
| 父文档检索 | ❌ | ✅ | ✅ |
| LLM 重排 | ❌ | ❌ | ✅ |
| Jina 重排 | ❌ | ❌ | ✅ |
| 重排采样数 | - | - | 12 |
| 检索 Top N | 10 | 10 | 5 |
| 并行请求数 | 10 | 20 | 4 |
| 默认模型 | gpt-4o-mini | gpt-4o | qwen-turbo |

#### 自定义配置示例

```python
from src.pipeline import Pipeline, RunConfig

custom_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    llm_reranking_sample_size=20,
    top_n_retrieval=8,
    parallel_requests=2,
    api_provider="dashscope",
    answering_model="qwen-turbo-latest",
    config_suffix="_custom"
)

pipeline = Pipeline(root_path, run_config=custom_config)
```

---

## 7. 运行方式

### 7.1 环境安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp env .env
# 编辑 .env 文件填入 API 密钥
```

### 7.2 CLI 命令行方式

**查看帮助**：
```bash
python main.py --help
```

**下载 Docling 模型**：
```bash
python main.py download-models
```

**解析 PDF**：
```bash
cd data/stock_data
python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
```

**序列化表格**：
```bash
python main.py serialize-tables --max-workers 5
```

**处理报告（分块+建库）**：
```bash
python main.py process-reports --config ser_tab
```

**处理问题**：
```bash
python main.py process-questions --config max
```

### 7.3 直接运行 pipeline.py

```bash
cd RAG-zdl
python src/pipeline.py
```

> 注意：需要先在 `pipeline.py` 底部取消对应方法的注释。

### 7.4 Streamlit Web UI

```bash
cd RAG-zdl
streamlit run app_streamlit.py
```

浏览器打开显示的地址（通常是 http://localhost:8501）。

### 7.5 Python API 调用

```python
from pathlib import Path
from src.pipeline import Pipeline, max_config

# 初始化
root_path = Path("data/stock_data")
pipeline = Pipeline(root_path, run_config=max_config)

# 单条问题提问
answer = pipeline.answer_single_question(
    "中芯国际2024年的营收是多少？",
    kind="number"
)
print(answer["final_answer"])

# 批量处理
pipeline.process_questions()
```

---

## 8. 依赖关系

### 8.1 模块依赖图

```
pipeline.py
    ├── pdf_parsing.py (PDFParser)
    ├── pdf_mineru.py (MinerU API)
    ├── parsed_reports_merging.py (PageTextPreparation)
    ├── text_splitter.py (TextSplitter)
    ├── ingestion.py (VectorDBIngestor, BM25Ingestor)
    ├── questions_processing.py (QuestionsProcessor)
    └── tables_serialization.py (TableSerializer)

questions_processing.py
    ├── retrieval.py (VectorRetriever, HybridRetriever)
    ├── api_requests.py (APIProcessor)
    └── prompts.py (提示词模板)

retrieval.py
    └── reranking.py (LLMReranker)

api_requests.py
    ├── prompts.py
    └── api_request_parallel_processor.py

tables_serialization.py
    └── api_requests.py (BaseOpenaiProcessor)
```

### 8.2 核心第三方依赖

| 库名 | 版本 | 用途 |
|------|------|------|
| docling | 2.14.0 | PDF 结构解析 |
| faiss-cpu | 1.9.0 | 向量相似度检索 |
| rank-bm25 | 0.2.2 | BM25 关键词检索 |
| langchain | 0.3.3 | 文本分块工具 |
| openai | 1.51.2 | OpenAI API 客户端 |
| dashscope | - | 阿里云 DashScope SDK |
| google-generativeai | 0.8.4 | Gemini API 客户端 |
| pydantic | 2.9.2 | 数据验证/结构化输出 |
| tiktoken | 0.8.0 | Token 计数 |
| streamlit | - | Web UI 框架 |
| click | 8.1.7 | CLI 框架 |
| pandas | 2.2.3 | 数据处理 |
| tenacity | - | 重试机制 |
| json_repair | 0.35.0 | JSON 修复 |

---

## 9. 关键类与函数索引

### 9.1 类索引

| 类名 | 所在文件 | 职责 |
|------|----------|------|
| `Pipeline` | [pipeline.py](src/pipeline.py#L68-L291) | 主流程编排 |
| `PipelineConfig` | [pipeline.py](src/pipeline.py#L22-L48) | 路径配置 |
| `RunConfig` | [pipeline.py](src/pipeline.py#L50-L67) | 运行配置 |
| `PDFParser` | [pdf_parsing.py](src/pdf_parsing.py#L32-L248) | PDF 解析 |
| `JsonReportProcessor` | [pdf_parsing.py](src/pdf_parsing.py#L250-L543) | 报告 JSON 组装 |
| `PageTextPreparation` | [parsed_reports_merging.py](src/parsed_reports_merging.py#L7-L435) | 报告文本规整 |
| `TextSplitter` | [text_splitter.py](src/text_splitter.py#L10-L203) | 文本分块 |
| `VectorDBIngestor` | [ingestion.py](src/ingestion.py#L56-L156) | 向量库构建 |
| `BM25Ingestor` | [ingestion.py](src/ingestion.py#L19-L54) | BM25 索引构建 |
| `VectorRetriever` | [retrieval.py](src/retrieval.py#L83-L283) | 向量检索 |
| `BM25Retriever` | [retrieval.py](src/retrieval.py#L19-L79) | BM25 检索 |
| `HybridRetriever` | [retrieval.py](src/retrieval.py#L286-L338) | 混合检索 |
| `LLMReranker` | [reranking.py](src/reranking.py#L38-L224) | LLM 重排 |
| `JinaReranker` | [reranking.py](src/reranking.py#L10-L35) | Jina 重排 |
| `QuestionsProcessor` | [questions_processing.py](src/questions_processing.py#L14-L572) | 问题处理 |
| `APIProcessor` | [api_requests.py](src/api_requests.py#L373-L518) | 统一 API 入口 |
| `BaseOpenaiProcessor` | [api_requests.py](src/api_requests.py#L20-L86) | OpenAI 处理器 |
| `BaseDashscopeProcessor` | [api_requests.py](src/api_requests.py#L673-L752) | DashScope 处理器 |
| `BaseIBMAPIProcessor` | [api_requests.py](src/api_requests.py#L89-L239) | IBM 处理器 |
| `BaseGeminiProcessor` | [api_requests.py](src/api_requests.py#L242-L371) | Gemini 处理器 |
| `AsyncOpenaiProcessor` | [api_requests.py](src/api_requests.py#L521-L671) | 异步 OpenAI 处理器 |
| `TableSerializer` | [tables_serialization.py](src/tables_serialization.py#L34-L303) | 表格序列化 |
| `TableSerialization` | [tables_serialization.py](src/tables_serialization.py#L306-L338) | 表格序列化提示词/Schema |

### 9.2 关键函数索引

| 函数名 | 所在文件/类 | 说明 |
|--------|-------------|------|
| `answer_single_question()` | [Pipeline](src/pipeline.py#L262-L291) | 单条问题即时推理 |
| `process_questions()` | [Pipeline](src/pipeline.py#L235-L260) | 批量处理问题 |
| `process_comparative_question()` | [QuestionsProcessor](src/questions_processing.py#L472-L537) | 多公司比较问题处理 |
| `get_answer_for_company()` | [QuestionsProcessor](src/questions_processing.py#L129-L178) | 单公司答案生成 |
| `rerank_documents()` | [LLMReranker](src/reranking.py#L146-L224) | 文档重排主方法 |
| `retrieve_by_company_name()` | [HybridRetriever](src/retrieval.py#L291-L338) | 混合检索 |
| `get_answer_from_rag_context()` | [APIProcessor](src/api_requests.py#L413-L461) | RAG 答案生成 |
| `get_rephrased_questions()` | [APIProcessor](src/api_requests.py#L503-L518) | 比较问题拆解 |
| `get_task_id()` | [pdf_mineru.py](src/pdf_mineru.py#L7-L25) | 提交 MinerU 解析任务 |
| `get_result()` | [pdf_mineru.py](src/pdf_mineru.py#L27-L68) | 获取 MinerU 解析结果 |
| `split_markdown_reports()` | [TextSplitter](src/text_splitter.py#L155-L203) | Markdown 报告批量分块 |

---

## 附录

### A. 数据格式说明

#### subset.csv 格式
| 列名 | 说明 |
|------|------|
| sha1 | PDF 文件 SHA1 哈希 |
| company_name | 公司名称 |
| file_name | 文件名（可选） |

#### questions.json 格式
```json
[
    {
        "text": "问题文本",
        "kind": "string"  // string, number, boolean, names
    }
]
```

#### 答案输出格式
```json
{
    "answers": [
        {
            "question_text": "问题文本",
            "kind": "string",
            "value": "答案内容",
            "references": [
                {
                    "pdf_sha1": "xxx",
                    "page_index": 0
                }
            ],
            "reasoning_process": "分步推理过程"
        }
    ],
    "details": "配置说明"
}
```

### B. 常见问题排查

1. **嵌入 API 返回空**
   - 检查 API 密钥是否正确
   - 查看 `embedding_error.log` 日志文件
   - 确认文本内容不为空

2. **DashScope QPS 超限**
   - 降低 `parallel_requests` 参数
   - 使用串行处理模式

3. **PDF 解析失败**
   - 检查 PDF 文件是否损坏
   - 尝试使用 MinerU 云端解析
   - 确保有足够的内存/GPU 资源

4. **答案质量不佳**
   - 尝试启用 `llm_reranking`
   - 调整 `top_n_retrieval` 数量
   - 启用 `parent_document_retrieval`
   - 检查 PDF 解析质量
