"""
GraphRAG Audit Knowledge Base - App Package
基于知识图谱的审计 RAG 知识库应用包。

业务入口为 ``app.main``（FastAPI 应用）。主要子包：
``core``（混合检索与生成）、``models``（API 与图谱 Schema 模型）、
``services``（Neo4j / 向量库）、``utils``（日志与提示词工具）。
"""

__version__ = "0.1.0"
__author__ = "Audit AI Team"
