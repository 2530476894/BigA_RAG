# graphrag-audit-kb

面向审计场景的 **混合检索（Neo4j + Chroma）+ 结构化 RAG 响应** 的最小可运行服务。当前生成阶段为**基于检索结果的占位回答**（未接入大语言模型）；检索与 API 闭环可独立验证。

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

## 目录说明

| 路径 | 说明 |
|------|------|
| `app/main.py` | FastAPI：`/health`、`/api/v1/rag/query`、图谱/向量统计与 `init-schema` |
| `app/config.py` | 环境变量与融合权重 |
| `app/core/retriever.py` | 向量 + 图谱混合检索 |
| `app/core/generator.py` | 无 LLM 时基于检索结果组装 `RAGQueryResponse` |
| `app/services/` | Neo4j、Chroma 封装 |
| `data/mock_triples.json` | 测试用三元组样例 |
| `tests/test_e2e_pipeline.py` | 环境与集成测试 |

## 常见问题

- **Neo4j/Chroma 未启动**：`/health` 为 `degraded`；检索可能为空，生成器会返回「未检索到相关知识」类提示。
- **Windows**：使用 `docker compose` 或 `docker-compose`；`run.sh` 需在 Bash 中执行。
