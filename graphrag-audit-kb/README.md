# graphrag-audit-kb

面向审计场景的 **LLM实体识别驱动的混合检索（Neo4j + Chroma）+ 结构化 RAG 响应** 的最小可运行服务。基于Qwen LLM实现审计领域实体智能提取、图谱语义链接和精准检索，显著提升审计知识库的问答准确性。

## 核心功能

- **LLM实体识别**：使用Qwen LLM从用户查询中智能提取审计领域实体（组织机构、法规条款、审计案例、风险事件），支持实体类型自动分类和置信度评估
- **图谱验证与链接**：将提取的实体与Neo4j知识图谱进行语义匹配验证，通过相似度计算和上下文关联实现实体到图谱节点的精准链接
- **混合检索增强**：结合向量相似度检索和图谱多跳关系查询，在实体识别基础上实现更智能的检索策略分配
- **结构化RAG响应**：基于检索结果生成专业审计咨询回答，包含完整的溯源路径（实体识别→向量检索→图谱检索）和合规建议

**技术优势**：
- 领域智能化：专门针对审计场景优化实体识别prompt和图谱模式
- 检索精确性：实体链接减少噪音，提高答案相关性和准确性
- 可解释性：完整的检索溯源，支持审计合规性验证
- 降级容错：LLM不可用时自动回退到传统关键词检索

## 最小闭环

1. **Python 3.10+**、**Docker**（用于 Neo4j 与 Chroma 容器）
2. 复制环境变量：`cp .env.example .env`（按需修改密码等）
3. 启动依赖：`docker compose up -d`（Neo4j `7474`/`7687`，Chroma `8000`）
4. 安装依赖：`pip install -r requirements.txt`
5. 初始化图约束（首次）：`curl -X POST http://localhost:8000/api/v1/graph/init-schema`
6. 写入测试数据：运行 `pytest tests/test_e2e_pipeline.py::TestMockDataIngestion -v`（在 Neo4j/Chroma 可用时将数据写入；否则仅校验 mock 文件）
7. 启动 API：`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
8. 验证：`pytest tests/ -v`，或访问 `http://localhost:8000/docs`

**一键脚本（Linux/macOS/Git Bash）**：`bash run.sh`（创建 venv、安装依赖、拉起 Docker；随后需手动执行第 5–7 步或 `uvicorn`）。

## API 示例

请求体字段为 **`question`**（不是 `query`）：

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"政府投资建设项目审计的主要法律依据是什么？\", \"vector_top_k\": 5, \"graph_hops\": 2}"
```

**响应字段说明**：
- `trace_paths`：新增 `entity_extraction` 路径类型，记录LLM识别的审计实体
- `confidence_score`：综合考虑实体识别置信度、检索结果质量的评分
- `validation_flags`：包含实体验证状态和图谱匹配信息

**实体识别健康检查**：
```bash
curl -X GET "http://localhost:8000/health/entity"
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `app/main.py` | FastAPI：`/health`、`/api/v1/rag/query`、图谱/向量统计与 `init-schema` |
| `app/config.py` | 环境变量、融合权重、实体识别/链接配置 |
| `app/core/retriever.py` | 向量 + 图谱混合检索 + LLM实体识别和语义匹配 |
| `app/core/generator.py` | 基于检索结果组装 `RAGQueryResponse`，包含实体溯源路径 |
| `app/services/llm_entity_service.py` | LLM实体提取和图谱链接服务 |
| `app/services/` | Neo4j、Chroma、LLM实体服务封装 |
| `data/mock_triples.json` | 测试用三元组样例 |
| `tests/test_e2e_pipeline.py` | 环境与集成测试（含实体识别测试） |

## 常见问题

- **Neo4j/Chroma 未启动**：`/health` 为 `degraded`；检索可能为空，生成器会返回「未检索到相关知识」类提示。
- **Windows**：使用 `docker compose` 或 `docker-compose`；`run.sh` 需在 Bash 中执行。
- **实体识别服务不可用**：`/health/entity` 返回错误；系统自动回退到关键词检索，响应中会标注 `validation_flags.fallback_used: true`。
- **实体置信度过低**：当 `confidence_score < 0.6` 时，建议检查查询表述或调整 `entity_extraction.min_confidence` 配置。
- **图谱链接失败**：实体识别成功但未找到图谱匹配；检查知识图谱数据完整性或调整 `entity_linking.similarity_threshold`。

## 配置说明

### 实体识别配置
```python
# app/config.py
entity_extraction:
  min_confidence: 0.6          # 实体识别最小置信度阈值
  max_entities: 5              # 单次查询最大提取实体数
  entity_types: ["组织机构", "法规条款", "审计案例", "风险事件"]  # 支持的实体类型
  prompt_template: "AUDIT_ENTITY_EXTRACTION_PROMPT"  # 使用的prompt模板
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
└─────────────────┘    └──────────────────┘             │
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  RAG Response   │◀───┤  Structured      │◀────────────┘
│  Generation     │    │  Answer Builder  │
└─────────────────┘    └──────────────────┘
```

## 技术栈

- **后端框架**：FastAPI (异步高性能API)
- **LLM服务**：Qwen LLM (通义千问，审计领域实体识别)
- **向量数据库**：Chroma (开源向量检索)
- **图数据库**：Neo4j (知识图谱存储和多跳查询)
- **配置管理**：Pydantic (类型安全配置)
- **日志系统**：structlog (结构化日志)
- **测试框架**：pytest (单元测试和集成测试)
- **容器化**：Docker Compose (Neo4j/Chroma服务编排)

## 开发说明

### 本地开发环境搭建
1. 克隆项目：`git clone <repository-url>`
2. 进入目录：`cd graphrag-audit-kb`
3. 配置环境：`cp .env.example .env`
4. 启动依赖服务：`docker compose up -d`
5. 安装Python依赖：`pip install -r requirements.txt`
6. 运行测试：`pytest tests/ -v`
7. 启动开发服务器：`uvicorn app.main:app --reload`

### 代码结构
```
graphrag-audit-kb/
├── app/                    # 应用核心代码
│   ├── main.py            # FastAPI应用入口
│   ├── config.py          # 配置管理
│   ├── core/              # 核心业务逻辑
│   │   ├── retriever.py   # 检索器
│   │   └── generator.py   # 生成器
│   ├── services/          # 服务层
│   │   ├── llm_entity_service.py  # LLM实体服务
│   │   ├── neo4j_service.py       # 图数据库服务
│   │   └── vector_service.py      # 向量数据库服务
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── data/                  # 数据文件
└── docker-compose.yml     # 容器编排配置
```

### 测试说明
- **单元测试**：`pytest tests/ -k "not e2e"`
- **集成测试**：`pytest tests/test_e2e_pipeline.py -v`
- **实体识别测试**：`pytest tests/test_e2e_pipeline.py::TestEntityRecognition -v`

## 部署说明

### Docker部署
```bash
# 构建镜像
docker build -t graphrag-audit-kb .

# 运行容器
docker run -p 8000:8000 --env-file .env graphrag-audit-kb
```

### 生产环境配置
- 设置环境变量：`QWEN_API_KEY`, `NEO4J_URI`, `CHROMA_HOST`
- 配置反向代理：Nginx或Caddy
- 监控健康检查：`/health` 和 `/health/entity` 端点
- 日志收集：配置structlog输出到文件或ELK栈

## 贡献指南

### 代码规范
- 使用 `black` 格式化代码：`black app/ tests/`
- 使用 `isort` 整理导入：`isort app/ tests/`
- 使用 `flake8` 检查代码质量：`flake8 app/ tests/`
- 提交前运行完整测试：`pytest tests/`

### 功能开发流程
1. 创建功能分支：`git checkout -b feature/entity-enhancement`
2. 编写测试用例
3. 实现功能代码
4. 运行测试验证：`pytest tests/ -v`
5. 提交代码：`git commit -m "feat: enhance entity recognition"`
6. 创建Pull Request

### 实体识别优化
- **Prompt工程**：在 `app/utils/prompts.py` 中优化 `AUDIT_ENTITY_EXTRACTION_PROMPT`
- **配置调优**：在 `app/config.py` 中调整实体识别和链接参数
- **算法改进**：在 `app/services/llm_entity_service.py` 中改进匹配算法
- **测试覆盖**：在 `tests/test_e2e_pipeline.py` 中添加新的测试用例

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- **项目维护者**：审计知识图谱团队
- **技术支持**：提交 GitHub Issue
- **文档更新**：欢迎提交 Pull Request

## 更新日志

### v1.1.0 (2024-12-XX)
- ✨ 新增LLM实体识别功能
  - 集成Qwen LLM进行审计领域实体提取
  - 支持实体类型自动分类和置信度评估
  - 实现实体到知识图谱的语义链接
- 🔧 增强混合检索策略
  - 基于实体识别结果动态调整检索权重
  - 新增实体相关性权重提升机制
  - 改进检索结果融合算法
- 📊 扩展API响应
  - 新增实体识别溯源路径
  - 添加综合置信度评分
  - 增加实体验证状态标记
- 🏥 新增健康检查
  - 实体识别服务健康检查端点
  - 降级容错机制
- 🧪 扩展测试覆盖
  - 实体识别准确性测试
  - 端到端RAG流程测试
- 📚 完善文档
  - 更新功能描述和技术架构
  - 添加配置参数说明
  - 补充常见问题解答

### v1.0.0 (2024-11-XX)
- 🎯 初始版本发布
- 🔍 基础混合检索功能（向量+图谱）
- 🤖 LLM生成回答
- 🐳 Docker容器化部署
- 📋 基础API接口

**核心组件说明**：
- **LLM Entity Extraction**：基于Qwen LLM的审计领域实体识别
- **Entity Linking & Validation**：实体到知识图谱的语义匹配和验证
- **Hybrid Retrieval Strategy**：根据实体识别结果动态调整检索策略
- **Result Fusion & Ranking**：加权融合向量和图谱检索结果
- **Structured Answer Builder**：生成包含溯源路径的专业审计回答
