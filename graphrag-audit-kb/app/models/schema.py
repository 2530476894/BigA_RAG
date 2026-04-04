"""
Pydantic Models - API 请求/响应与 RAG 结构化输出
"""

from typing import List, Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    """风险等级"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RAGQueryRequest(BaseModel):
    """RAG 查询请求模型"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "政府投资建设项目审计的主要法律依据是什么？",
                "vector_top_k": 5,
                "graph_hops": 2,
                "include_cases": True,
                "include_regulations": True,
            }
        }
    )

    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    vector_top_k: int = Field(default=5, ge=1, le=20, description="向量检索 TopK")
    graph_hops: int = Field(default=2, ge=1, le=5, description="图谱跳跃层数")
    include_cases: bool = Field(default=True, description="是否包含关联案例")
    include_regulations: bool = Field(default=True, description="是否包含相关法规")


class BasisClause(BaseModel):
    """依据条款模型（RAG 响应用）"""

    clause_id: str = Field(..., description="条款编号")
    clause_content: str = Field(..., description="条款原文摘要")
    source: str = Field(..., description="法规/制度名称")
    effectiveness_level: str = Field(..., description="效力等级")


class RelatedCase(BaseModel):
    """关联案例模型（RAG 响应用）"""

    case_id: str = Field(..., description="案例编号")
    case_summary: str = Field(..., description="案例摘要")
    similarity_score: float = Field(..., description="相似度分数", ge=0.0, le=1.0)
    outcome: Optional[str] = Field(default=None, description="处理结果")


class TracePath(BaseModel):
    """溯源路径模型（RAG 响应用）"""

    path_type: Literal["vector", "graph"] = Field(..., description="路径类型")
    path_description: str = Field(..., description="溯源路径描述")
    nodes: List[str] = Field(default_factory=list, description="路径节点列表")


class ValidationFlags(BaseModel):
    """校验标志模型（RAG 响应用）"""

    amount_validated: bool = Field(..., description="金额是否已校验")
    time_validated: bool = Field(..., description="时间是否已校验")
    uncertainty_notes: List[str] = Field(default_factory=list, description="不确定性说明")


class RAGQueryResponse(BaseModel):
    """RAG 查询响应模型"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "政府投资建设项目审计的主要法律依据包括《中华人民共和国审计法》...",
                "basis_clauses": [],
                "related_cases": [],
                "confidence_score": 0.85,
                "trace_paths": [],
                "validation_flags": {
                    "amount_validated": True,
                    "time_validated": True,
                    "uncertainty_notes": [],
                },
                "risk_level": "low",
                "compliance_suggestions": ["建议查阅法规原文"],
            }
        }
    )

    answer: str = Field(..., description="核心结论")
    basis_clauses: List[BasisClause] = Field(default_factory=list, description="依据条款列表")
    related_cases: List[RelatedCase] = Field(default_factory=list, description="关联案例列表")
    confidence_score: float = Field(..., description="置信度", ge=0.0, le=1.0)
    trace_paths: List[TracePath] = Field(default_factory=list, description="溯源路径列表")
    validation_flags: ValidationFlags = Field(..., description="校验标志")
    risk_level: RiskLevel = Field(..., description="风险等级")
    compliance_suggestions: List[str] = Field(default_factory=list, description="合规建议列表")
