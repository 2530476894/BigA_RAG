"""
Prompts Module - 审计检索上下文格式化与 RAG Prompt 模板

用途：提供审计场景的 Prompt 模板和上下文格式化函数
关键依赖：无（纯 Python）
"""

from datetime import datetime
from typing import Any, Dict, List


# ==================== System Prompts ====================

AUDIT_RAG_SYSTEM_PROMPT = """你是一名专业的审计知识库助手，基于检索到的法规条款、案例和风险事件为用户提供准确的审计咨询回答。

请遵循以下原则：
1. **准确性优先**：严格基于提供的检索上下文回答，不要编造信息
2. **引用来源**：回答中应明确指出引用的法规条款或案例来源
3. **结构化输出**：使用清晰的段落和要点组织回答
4. **风险提示**：如涉及风险事件，应说明风险等级和合规建议
5. **不确定性处理**：如检索信息不足，应明确告知用户并建议查阅原文

如果检索上下文中没有相关信息，请如实告知用户，不要强行回答。
"""


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


def build_rag_prompt(
    question: str,
    vector_results: List[dict],
    graph_results: List[dict],
    vector_top_k: int = 5,
    graph_hops: int = 2,
) -> str:
    """
    构建完整的 RAG Prompt，包含系统指令和检索上下文
    
    Args:
        question: 用户问题
        vector_results: 向量检索结果
        graph_results: 图谱检索结果
        vector_top_k: 向量结果展示数量
        graph_hops: 图谱检索跳数
        
    Returns:
        完整的 Prompt 字符串
    """
    context = format_audit_context(
        question=question,
        vector_results=vector_results,
        graph_results=graph_results,
        vector_top_k=vector_top_k,
        graph_hops=graph_hops,
    )
    
    prompt = f"""请基于以下检索到的审计知识回答用户问题。

【检索上下文】
===== 向量检索结果（相关文档片段） =====
{context['vector_context']}

===== 图谱检索结果（关联实体与关系） =====
{context['graph_context']}

===== 用户问题 =====
{question}

请根据上述检索内容，给出专业、准确的审计咨询回答。回答时应：
1. 引用具体的法规条款或案例来源
2. 如适用，说明风险等级和合规建议
3. 保持回答结构清晰、易于理解

回答："""
    
    return prompt
