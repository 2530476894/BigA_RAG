"""
Prompts Module - 审计专用 Prompt 模板

用途：定义 RAG 生成器的 Prompt 模板，强制审计合规约束
关键依赖：langchain core prompts
审计场景映射：法规效力优先、可追溯性、金额/时间敏感、合规裁量提示
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ==================== System Prompt ====================
# 审计 RAG 系统指令：强制输出格式与合规约束
AUDIT_SYSTEM_PROMPT = """你是一个专业的审计合规助手，基于检索到的知识图谱和文档片段回答用户问题。

【核心原则】
1. 法规效力优先：优先引用国家法律法规、部门规章，其次地方规定，最后企业内部制度
2. 可追溯性：所有结论必须有明确的条款或案例支撑，禁止无依据推断
3. 金额/时间敏感：涉及金额、日期、期限的内容需二次校验，标注不确定性
4. 合规裁量提示：对于存在裁量空间的事项，明确说明裁量因素和风险等级

【输出格式要求】
你必须严格按照以下 JSON 格式输出，不得遗漏任何字段：
{{
    "answer": "直接回答用户问题的核心结论",
    "basis_clauses": [
        {{
            "clause_id": "条款编号",
            "clause_content": "条款原文摘要",
            "source": "法规/制度名称",
            "effectiveness_level": "法律/行政法规/部门规章/地方规定/内部制度"
        }}
    ],
    "related_cases": [
        {{
            "case_id": "案例编号",
            "case_summary": "案例摘要",
            "similarity_score": 0.95,
            "outcome": "处理结果"
        }}
    ],
    "confidence_score": 0.85,
    "trace_paths": [
        {{
            "path_type": "vector/graph",
            "path_description": "溯源路径描述",
            "nodes": ["节点 1", "节点 2"]
        }}
    ],
    "validation_flags": {{
        "amount_validated": true,
        "time_validated": true,
        "uncertainty_notes": []
    }},
    "risk_level": "low/medium/high",
    "compliance_suggestions": ["合规建议 1", "合规建议 2"]
}}

【禁止事项】
- 禁止编造不存在的法规条款或案例
- 禁止对未检索到的信息进行推测
- 禁止忽略金额、时间的不确定性
- 禁止给出绝对化的合规结论（除非有明确法律依据）

如果检索结果为空或不足以支撑回答，请如实说明，并建议补充查询方向。"""


# ==================== Human Prompt Template ====================
# 用户问题 + 检索上下文的组合模板
AUDIT_HUMAN_PROMPT = """【用户问题】
{question}

【向量检索结果】(Top {vector_top_k}, 权重：{vector_weight})
{vector_context}

【知识图谱检索结果】({graph_hops}跳，权重：{graph_weight})
{graph_context}

【当前时间】{current_time}

请根据上述检索结果，按照系统指令要求的格式回答问题。
如果某些字段无法从检索结果中获取，请在对应字段标注"未检索到相关信息"，并在 uncertainty_notes 中说明。"""


# ==================== Entity Extraction Prompt ====================
# 用于从审计文档中抽取实体和关系的 Prompt
ENTITY_EXTRACTION_PROMPT = """你是一个审计领域的信息抽取专家。请从以下文本中抽取审计相关的实体和关系。

【待处理文本】
{text}

【抽取要求】
1. 实体类型包括：Organization(组织), Regulation(法规), AuditCase(审计案例), RiskEvent(风险事件), Person(人员), Amount(金额), Time(时间)
2. 关系类型包括：
   - ISSUED_BY(法规由...发布)
   - APPLIES_TO(法规适用于...)
   - INVOLVED_IN(组织卷入...案例)
   - DETECTED_IN(风险事件在...中发现)
   - VIOLATED(违反...法规)
   - PENALIZED_BY(被...处罚)
   - OCCURRED_AT(发生于...时间)
   - AMOUNT_INVOLVED(涉及金额...)

【输出格式】
请以 JSON 格式输出，结构如下：
{{
    "entities": [
        {{"type": "Organization", "name": "XX 公司", "attributes": {{"industry": "建筑业"}}}}
    ],
    "relations": [
        {{"source": "XX 公司", "target": "YY 违规案", "type": "INVOLVED_IN"}}
    ]
}}

只输出 JSON，不要有其他内容。"""


# ==================== Prompt Templates ====================
# LangChain ChatPromptTemplate 实例
audit_rag_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(AUDIT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(AUDIT_HUMAN_PROMPT),
])

entity_extraction_prompt = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT)


def format_audit_context(
    question: str,
    vector_results: list[dict],
    graph_results: list[dict],
    vector_top_k: int = 5,
    graph_hops: int = 2,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
    current_time: str = ""
) -> dict:
    """
    格式化审计检索上下文
    
    Args:
        question: 用户问题
        vector_results: 向量检索结果列表
        graph_results: 图谱检索结果列表
        vector_top_k: 向量 TopK 参数
        graph_hops: 图谱跳跃层数
        vector_weight: 向量权重
        graph_weight: 图谱权重
        current_time: 当前时间戳
        
    Returns:
        格式化后的 prompt 输入字典
    """
    from datetime import datetime
    
    # 格式化向量检索结果
    vector_context_parts = []
    for i, result in enumerate(vector_results[:vector_top_k], 1):
        chunk = result.get("chunk", "")
        source = result.get("source", "未知来源")
        score = result.get("score", 0.0)
        vector_context_parts.append(
            f"[{i}] 来源：{source} | 相似度：{score:.3f}\n    内容：{chunk}"
        )
    vector_context = "\n\n".join(vector_context_parts) if vector_context_parts else "未检索到相关文档片段"
    
    # 格式化图谱检索结果
    graph_context_parts = []
    for i, result in enumerate(graph_results, 1):
        path_desc = result.get("path_description", "")
        nodes = result.get("nodes", [])
        properties = result.get("properties", {})
        graph_context_parts.append(
            f"[{i}] 路径：{path_desc}\n    节点：{' -> '.join(nodes)}\n    属性：{properties}"
        )
    graph_context = "\n\n".join(graph_context_parts) if graph_context_parts else "未检索到关联图谱数据"
    
    return {
        "question": question,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "vector_top_k": vector_top_k,
        "graph_hops": graph_hops,
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "current_time": current_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
