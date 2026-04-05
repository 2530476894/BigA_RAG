"""
Core Package - 核心业务逻辑包

本文件仅聚合导出子模块中的符号；检索与生成实现见 ``retriever``、``generator``。
"""

from app.core.retriever import (
    HybridRetriever,
    get_hybrid_retriever,
)
from app.core.generator import (
    RAGGenerator,
    get_generator,
)

__all__ = [
    "HybridRetriever",
    "get_hybrid_retriever",
    "RAGGenerator",
    "get_generator",
]
