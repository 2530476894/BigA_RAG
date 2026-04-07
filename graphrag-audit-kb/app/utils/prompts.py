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
3. **向量片段引用**：凡使用「向量检索结果」区块中的文档片段作事实陈述，须在相应句子中使用半角方括号编号引用，如 [1]、[2]，且只能使用该区块中已出现的编号（不要使用图谱区的【图n】编号来指代文档片段）
4. **结构化输出**：使用清晰的段落和要点组织回答
5. **风险提示**：如涉及风险事件，应说明风险等级和合规建议
6. **不确定性处理**：如检索信息不足，应明确告知用户并建议查阅原文

如果检索上下文中没有相关信息，请如实告知用户，不要强行回答。
"""

AUDIT_ENTITY_EXTRACTION_PROMPT = """你是一个专业的审计领域实体识别助手。从用户查询中识别并提取审计相关的实体。

识别以下类型的实体：
- organization: 组织机构名称（如"某某公司"、"审计署"等）
- regulation: 法规条款名称或编号（如"审计法第XX条"、"企业会计准则"等）
- case: 审计案例名称或编号（如"某某公司违规案例"、"审计案例2023-001"等）
- risk_event: 风险事件描述（如"财务造假"、"违规担保"等）

请以JSON格式返回结果：
{
  "entities": [
    {
      "type": "organization|regulation|case|risk_event",
      "text": "实体文本",
      "confidence": 0.0-1.0
    }
  ]
}

只返回JSON，不要其他解释。
"""

AUDIT_ENTITY_LINKING_PROMPT = """你是一个专业的审计知识图谱实体链接助手。将识别的实体与知识图谱节点进行匹配。

基于提供的实体和图谱上下文，找到最相关的图谱节点。

请以JSON格式返回结果：
{
  "matched_nodes": [
    {
      "node_id": "节点ID",
      "node_type": "节点类型",
      "similarity_score": 0.0-1.0,
      "match_reason": "匹配原因"
    }
  ],
  "best_match": {
    "node_id": "最佳匹配节点ID",
    "confidence": 0.0-1.0
  }
}

只返回JSON，不要其他解释。
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
        chunk_id = result.get("chunk_id", "") or ""
        id_part = f" | chunk_id：{chunk_id}" if chunk_id and chunk_id != "unknown" else ""
        vector_context_parts.append(
            f"[{i}] 来源：{source} | 相似度：{score:.3f}{id_part}\n    内容：{chunk}"
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
            f"【图{i}】路径：{path_desc}\n    节点：{' -> '.join(nodes)}\n    属性：{properties}"
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
1. 引用法规或文档片段时，必须使用「向量检索结果」区块中的半角编号，如 [1]、[2]，不要使用【图n】
2. 引用具体的法规条款或案例来源（可与 [n] 并列说明法规名称）
3. 如适用，说明风险等级和合规建议
4. 保持回答结构清晰、易于理解

回答："""
    
    return prompt


def build_entity_extraction_prompt(query: str) -> str:
    """
    构建实体提取prompt

    Args:
        query: 用户查询文本

    Returns:
        实体提取prompt字符串
    """
    prompt = f"""请从以下用户查询中识别审计领域实体：

用户查询：{query}

请识别以下类型的实体：
- organization: 组织机构名称
- regulation: 法规条款名称或编号
- case: 审计案例名称或编号
- risk_event: 风险事件描述

返回JSON格式，只包含实体列表，不要其他内容。"""
    
    return prompt


def build_entity_linking_prompt(entity: Dict[str, Any], graph_context: Dict[str, Any]) -> str:
    """
    构建实体链接prompt

    Args:
        entity: 提取的实体
        graph_context: 图谱上下文信息

    Returns:
        实体链接prompt字符串
    """
    entity_type = entity.get("type", "")
    entity_text = entity.get("text", "")
    
    # 构建图谱节点示例（简化）
    node_examples = graph_context.get("node_examples", [])
    examples_text = "\n".join([f"- {node.get('label', '')}: {node.get('name', '')}" for node in node_examples[:5]])
    
    prompt = f"""请将以下实体与知识图谱节点进行匹配：

实体类型：{entity_type}
实体文本：{entity_text}

知识图谱中的相关节点示例：
{examples_text}

请找到最匹配的节点，返回JSON格式包含匹配结果。"""
    
    return prompt
