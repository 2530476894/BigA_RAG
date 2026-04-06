# BigA_RAG - 审计领域知识图谱问答系统

面向审计场景的 **LLM 实体识别驱动的混合检索（Neo4j + Chroma）+ 结构化 RAG 响应** 服务。基于阿里云通义千问（Qwen）实现审计领域实体智能提取、图谱语义链接和精准检索，显著提升审计知识库的问答准确性。

## 项目简介

本项目是一个面向审计场景的智能问答系统，结合图数据库（Neo4j）和向量数据库（Chroma），实现混合检索增强生成（RAG）。通过 LLM 实体识别、图谱验证与链接、混合检索增强和结构化响应生成，提供专业、可溯源的审计咨询回答。

## 核心特性

- 🧠 **LLM 实体识别**：使用 Qwen LLM 从用户查询中智能提取审计领域实体（组织机构、法规条款、审计案例、风险事件），支持实体类型自动分类和置信度评估
- � **图谱验证与链接**：将提取的实体与 Neo4j 知识图谱进行语义匹配验证，通过相似度计算和上下文关联实现实体到图谱节点的精准链接
- 🔍 **混合检索增强**：结合向量相似度检索（Chroma）和图谱多跳关系查询（Neo4j），在实体识别基础上实现更智能的检索策略分配
- 📊 **结构化 RAG 响应**：基于检索结果生成专业审计咨询回答，包含完整的溯源路径（实体识别→向量检索→图谱检索）和合规建议
- 🛡️ **降级容错**：LLM 不可用时自动回退到传统关键词检索，响应中标注 `validation_flags.fallback_used: true`
- 🐳 **容器化部署**：支持 Docker Compose 一键启动依赖服务

**技术优势**：
- **领域智能化**：专门针对审计场景优化实体识别 prompt 和图谱模式
- **检索精确性**：实体链接减少噪音，提高答案相关性和准确性
- **可解释性**：完整的检索溯源，支持审计合规性验证
- **降级容错**：LLM 不可用时自动回退到传统关键词检索

## 架构图

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  LLM Entity      │───▶│  Entity Linking │
│                 │    │  Extraction      │    │  & Validation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  Vector Search  │◀───┤  Hybrid Retrieval│◀────────────┘
│  (Chroma)       │    │  Strategy        │
└─────────────────┘    └──────────────────┘             │
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  Graph Query    │◀───┤  Result Fusion   │◀────────────┘
│  (Neo4j)        │    │  & Ranking       │
└─────────────────┘    └─────────────────┘             │
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  RAG Response   │◀───┤  Structured      │◀────────────┘
│  Generation     │    │  Answer Builder  │
└─────────────────┘    └─────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose

### 安装步骤

```bash
# 进入项目目录
cd graphrag-audit-kb

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 文件，填入你的 DashScope API Key
# DASHSCOPE_API_KEY=your_api_key_here

# 启动依赖服务（Neo4j + Chroma）
docker compose up -d

# 安装 Python 依赖
pip install -r requirements.txt

# 初始化图谱约束（首次运行）
curl -X POST http://localhost:8000/api/v1/graph/init-schema

# 写入测试数据（可选）
pytest tests/test_e2e_pipeline.py::TestMockDataIngestion -v

# 启动 API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 一键脚本（Linux/macOS/Git Bash）

```bash
cd graphrag-audit-kb
bash run.sh
```

> **注意**：`run.sh` 会创建 venv、安装依赖、拉起 Docker；随后需手动执行图谱初始化和启动 API 服务。

## API 使用示例

### 健康检查

```bash
# 系统健康检查
curl http://localhost:8000/health

# 实体识别健康检查
curl http://localhost:8000/health/entity
```

### RAG 查询

请求体字段为 **`question`**（不是 `query`）：

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "政府投资建设项目审计的主要法律依据是什么？",
    "vector_top_k": 5,
    "graph_hops": 2
  }'
```

### 响应字段说明

| 字段 | 说明 |
|------|------|
| `answer` | 生成的审计咨询回答 |
| `trace_paths` | 检索溯源路径，新增 `entity_extraction` 类型记录 LLM 识别的审计实体 |
| `confidence_score` | 综合考虑实体识别置信度、检索结果质量的评分 |
| `validation_flags` | 包含实体验证状态和图谱匹配信息，如 `fallback_used` |

### 访问交互式文档

浏览器打开：http://localhost:8000/docs

## 项目结构

```
BigA_RAG/
├── README.md                         # 本文件
└── graphrag-audit-kb/                # 主项目目录
    ├── app/                          # 应用核心代码
    │   ├── main.py                   # FastAPI 入口，/health、/api/v1/rag/query 等
    │   ├── config.py                 # 配置管理（环境变量、融合权重、实体识别/链接配置）
    │   ├── core/                     # 核心业务逻辑
    │   │   ├── retriever.py          # 混合检索器（向量 + 图谱 + LLM 实体识别）
    │   │   └── generator.py          # 生成器（组装 RAGQueryResponse，包含实体溯源）
    │   ├── services/                 # 服务层
    │   │   ├── llm_entity_service.py # LLM 实体提取和图谱链接服务
    │   │   ├── neo4j_service.py      # 图数据库服务
    │   │   └── vector_service.py     # 向量数据库服务
    │   ├── llm/                      # LLM 抽象层
    │   │   └── __init__.py           # Qwen 实现
    │   ├── models/                   # 数据模型
    │   │   ├── schema.py             # API 请求/响应模型
    │   │   └── kg_schema.py          # 知识图谱模式
    │   └── utils/                    # 工具函数
    │       ├── prompts.py            # Prompt 模板
    │       └── logger.py             # 结构化日志
    ├── data/                         # 数据文件
    │   ├── mock/mock_triples.json    # 测试用三元组样例
    │   └── sample/                   # 样例数据
    │       ├── audit_case_sample.json
    │       └── audit_regulations_sample.txt
    ├── tests/                        # 测试代码
    │   └── test_e2e_pipeline.py      # 环境与集成测试（含实体识别测试）
    ├── docker-compose.yml            # Docker 配置
    ├── requirements.txt              # Python 依赖
    └── run.sh                        # 一键启动脚本
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Web 框架 | FastAPI（异步高性能 API） |
| 图数据库 | Neo4j（知识图谱存储和多跳查询） |
| 向量数据库 | Chroma（开源向量检索） |
| LLM | Qwen（通义千问，审计领域实体识别） |
| Embedding | text-embedding-v3 |
| 配置管理 | Pydantic（类型安全配置） |
| 日志系统 | structlog（结构化日志） |
| 测试框架 | pytest（单元测试和集成测试） |
| 容器化 | Docker Compose（Neo4j/Chroma 服务编排） |

## 配置说明

### 实体识别配置

```python
# app/config.py
entity_extraction:
  min_confidence: 0.6          # 实体识别最小置信度阈值
  max_entities: 5              # 单次查询最大提取实体数
  entity_types: ["组织机构", "法规条款", "审计案例", "风险事件"]  # 支持的实体类型
  prompt_template: "AUDIT_ENTITY_EXTRACTION_PROMPT"  # 使用的 prompt 模板
```

### 实体链接配置

```python
entity_linking:
  similarity_threshold: 0.8     # 图谱节点匹配相似度阈值
  max_candidates: 3             # 每个实体最大候选匹配数
  semantic_search_enabled: true # 是否启用语义搜索
  fallback_to_keyword: true     # 匹配失败时是否回退到关键词搜索
```

### 混合检索配置

```python
retrieval:
  vector_weight: 0.6            # 向量检索权重
  graph_weight: 0.4             # 图谱检索权重
  entity_boost: 1.2             # 实体相关结果的权重提升
  fusion_strategy: "weighted"   # 检索结果融合策略
```

## 模型选择

### Embedding 模型
- **text-embedding-v3**（推荐）：支持中英文，维度 1536，适用于语义相似度检索

### LLM 模型
- **qwen-max**：最强推理能力，适合复杂分析
- **qwen-plus**（推荐）：平衡性能与成本
- **qwen-turbo**：快速响应，适合简单问答

## 获取 API Key

访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/apiKey) 获取 API Key。

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行端到端测试
pytest tests/test_e2e_pipeline.py -v

# 运行实体识别测试
pytest tests/test_e2e_pipeline.py::TestEntityRecognition -v

# 运行单元测试（跳过 e2e）
pytest tests/ -k "not e2e"
```

## 开发说明

### 本地开发环境搭建

1. 克隆项目：`git clone <repository-url>`
2. 进入目录：`cd graphrag-audit-kb`
3. 配置环境：`cp .env.example .env`
4. 启动依赖服务：`docker compose up -d`
5. 安装 Python 依赖：`pip install -r requirements.txt`
6. 运行测试：`pytest tests/ -v`
7. 启动开发服务器：`uvicorn app.main:app --reload`

### 代码规范

- 使用 `black` 格式化代码：`black app/ tests/`
- 使用 `isort` 整理导入：`isort app/ tests/`
- 使用 `flake8` 检查代码质量：`flake8 app/ tests/`
- 提交前运行完整测试：`pytest tests/`

### 功能开发流程

1. 创建功能分支：`git checkout -b feature/your-feature`
2. 编写测试用例
3. 实现功能代码
4. 运行测试验证：`pytest tests/ -v`
5. 提交代码：`git commit -m "feat: your feature description"`
6. 创建 Pull Request

### 实体识别优化

- **Prompt 工程**：在 `app/utils/prompts.py` 中优化 `AUDIT_ENTITY_EXTRACTION_PROMPT`
- **配置调优**：在 `app/config.py` 中调整实体识别和链接参数
- **算法改进**：在 `app/services/llm_entity_service.py` 中改进匹配算法
- **测试覆盖**：在 `tests/test_e2e_pipeline.py` 中添加新的测试用例

## 部署说明

### Docker 部署

```bash
# 构建镜像
docker build -t graphrag-audit-kb .

# 运行容器
docker run -p 8000:8000 --env-file .env graphrag-audit-kb
```

### 生产环境配置

- 设置环境变量：`DASHSCOPE_API_KEY`, `NEO4J_URI`, `CHROMA_HOST`
- 配置反向代理：Nginx 或 Caddy
- 监控健康检查：`/health` 和 `/health/entity` 端点
- 日志收集：配置 structlog 输出到文件或 ELK 栈

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| **Neo4j/Chroma 未启动** | `/health` 为 `degraded`；检索可能为空，生成器会返回「未检索到相关知识」类提示 |
| **Windows 用户** | 使用 `docker compose` 或 `docker-compose`；`run.sh` 需要在 Git Bash 中执行 |
| **实体识别服务不可用** | `/health/entity` 返回错误；系统自动回退到关键词检索，响应中会标注 `validation_flags.fallback_used: true` |
| **实体置信度过低** | 当 `confidence_score < 0.6` 时，建议检查查询表述或调整 `entity_extraction.min_confidence` 配置 |
| **图谱链接失败** | 实体识别成功但未找到图谱匹配；检查知识图谱数据完整性或调整 `entity_linking.similarity_threshold` |
| **检索结果为空** | 1. 确认依赖服务已启动；2. 检查是否已写入测试数据；3. 确认 API Key 配置正确 |

## 更新日志

### v1.1.0 (2024-12-XX)
- ✨ 新增 LLM 实体识别功能
  - 集成 Qwen LLM 进行审计领域实体提取
  - 支持实体类型自动分类和置信度评估
  - 实现实体到知识图谱的语义链接
- 🔧 增强混合检索策略
  - 基于实体识别结果动态调整检索权重
  - 新增实体相关性权重提升机制
  - 改进检索结果融合算法
- 📊 扩展 API 响应
  - 新增实体识别溯源路径
  - 添加综合置信度评分
  - 增加实体验证状态标记
- 🏥 新增健康检查
  - 实体识别服务健康检查端点
  - 降级容错机制
- 🧪 扩展测试覆盖
  - 实体识别准确性测试
  - 端到端 RAG 流程测试
- 📚 完善文档
  - 更新功能描述和技术架构
  - 添加配置参数说明
  - 补充常见问题解答

### v1.0.0 (2024-11-XX)
- 🎯 初始版本发布
- 🔍 基础混合检索功能（向量+图谱）
- 🤖 LLM 生成回答
- 🐳 Docker 容器化部署
- 📋 基础 API 接口

## 许可证

MIT License

## 联系方式

- **项目维护者**：审计知识图谱团队
- **技术支持**：提交 GitHub Issue
- **文档更新**：欢迎提交 Pull Request
