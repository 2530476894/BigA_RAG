"""
Prompts Module - 审计检索上下文格式化（纯 Python，无 LangChain 依赖）
"""

from datetime import datetime
from typing import Any, Dict, List


def format_audit_context(
    question: str,
    vector_results: List[dict],
    graph_results: List[dict],
    vector_top_k: int = 5,
    graph_hops: int = 2,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
    current_time: str = "",
) -> Dict[str, Any]:
    """
    将向量与图谱检索结果格式化为可读上下文，供生成器拼装 answer。

    Returns:
        字典键：``question``；``vector_context`` / ``graph_context`` 为两侧可读文本；
        ``vector_top_k``、``graph_hops``、``vector_weight``、``graph_weight`` 为参数回显；
        ``current_time`` 为时间戳字符串（可由参数传入覆盖）。
    """
    vector_context_parts = []
    for i, result in enumerate(vector_results[:vector_top_k], 1):
        chunk = result.get("chunk", "")
        source = result.get("source", "未知来源")
        score = result.get("score", 0.0)
        vector_context_parts.append(
            f"[{i}] 来源：{source} | 相似度：{score:.3f}\n    内容：{chunk}"
        )
    vector_context = (
        "\n\n".join(vector_context_parts)
        if vector_context_parts
        else "未检索到相关文档片段"
    )

    graph_context_parts = []
    for i, result in enumerate(graph_results, 1):
        path_desc = result.get("path_description", "")
        nodes = result.get("nodes", [])
        properties = result.get("properties", {})
        graph_context_parts.append(
            f"[{i}] 路径：{path_desc}\n    节点：{' -> '.join(nodes)}\n    属性：{properties}"
        )
    graph_context = (
        "\n\n".join(graph_context_parts)
        if graph_context_parts
        else "未检索到关联图谱数据"
    )

    return {
        "question": question,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "vector_top_k": vector_top_k,
        "graph_hops": graph_hops,
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "current_time": current_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
