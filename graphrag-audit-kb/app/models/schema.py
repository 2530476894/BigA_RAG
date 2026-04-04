"""
Pydantic Models - 数据校验模型

用途：定义 API 请求/响应模型和业务实体
关键依赖：pydantic
审计场景映射：审计案例、法规条款、风险事件、组织机构等核心实体
"""

from datetime import datetime
from typing import Optional, List, Literal
import pydantic
from pydantic import BaseModel, Field


# ==================== Effectiveness Level Enum ====================
class EffectivenessLevel(str):
    """法规效力等级"""
    LAW = "法律"
    ADMINISTRATIVE_REGULATION = "行政法规"
    DEPARTMENT_RULE = "部门规章"
    LOCAL_REGULATION = "地方规定"
    INTERNAL_POLICY = "内部制度"


# ==================== Risk Level Enum ====================
class RiskLevel(str):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ==================== Core Entity Models ====================
class Organization(BaseModel):
    """
    组织机构模型
    审计场景：被审计单位、监管机构、中介机构等
    """
    id: str = Field(..., description="组织唯一标识")
    name: str = Field(..., description="组织名称")
    org_type: str = Field(default="", description="组织类型：政府机关/企业/事业单位/社会团体")
    industry: Optional[str] = Field(default=None, description="所属行业")
    registration_code: Optional[str] = Field(default=None, description="统一社会信用代码")
    attributes: Optional[dict] = Field(default=None, description="扩展属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "org_001",
                "name": "XX 建设集团有限公司",
                "org_type": "企业",
                "industry": "建筑业",
                "registration_code": "91110000XXXXXXXXXX"
            }
        }


class Regulation(BaseModel):
    """
    法规条款模型
    审计场景：法律法规、部门规章、地方规定、企业内部制度
    """
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "reg_001",
                "title": "中华人民共和国审计法",
                "clause_id": "第二十二条",
                "clause_content": "审计机关对政府投资和以政府投资为主的建设项目的预算执行情况和决算进行审计监督。",
                "effectiveness_level": "法律",
                "issuing_authority": "全国人民代表大会常务委员会",
                "effective_date": "2022-01-01T00:00:00",
                "status": "effective"
            }
        }
    )
    
    id: str = Field(..., description="条款唯一标识")
    title: str = Field(..., description="法规标题")
    clause_id: Optional[str] = Field(default=None, description="具体条款编号")
    clause_content: str = Field(..., description="条款原文内容")
    effectiveness_level: EffectivenessLevel = Field(..., description="效力等级")
    issuing_authority: Optional[str] = Field(default=None, description="发布机关")
    effective_date: Optional[datetime] = Field(default=None, description="生效日期")
    status: str = Field(default="effective", description="状态：effective/abolished/amended")
    attributes: Optional[dict] = Field(default=None, description="扩展属性")


class AuditCase(BaseModel):
    """
    审计案例模型
    审计场景：历史审计项目、违规案例、处罚案例
    """
    id: str = Field(..., description="案例唯一标识")
    case_number: Optional[str] = Field(default=None, description="案例编号")
    title: str = Field(..., description="案例标题")
    summary: str = Field(..., description="案例摘要")
    involved_orgs: List[str] = Field(default_factory=list, description="涉及组织 ID 列表")
    violation_types: List[str] = Field(default_factory=list, description="违规类型列表")
    amount_involved: Optional[float] = Field(default=None, description="涉及金额")
    penalty_amount: Optional[float] = Field(default=None, description="处罚金额")
    outcome: Optional[str] = Field(default=None, description="处理结果")
    case_date: Optional[datetime] = Field(default=None, description="案例发生日期")
    attributes: Optional[dict] = Field(default=None, description="扩展属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "case_001",
                "case_number": "审罚〔2023〕001 号",
                "title": "XX 公司虚增利润案",
                "summary": "该公司通过虚构交易、提前确认收入等方式虚增利润...",
                "involved_orgs": ["org_001"],
                "violation_types": ["财务造假", "信息披露违规"],
                "amount_involved": 50000000.0,
                "penalty_amount": 5000000.0,
                "outcome": "责令改正，罚款 500 万元",
                "case_date": "2023-06-15T00:00:00"
            }
        }


class RiskEvent(BaseModel):
    """
    风险事件模型
    审计场景：审计发现的风险点、异常事项
    """
    id: str = Field(..., description="风险事件唯一标识")
    event_type: str = Field(..., description="风险类型：财务/合规/运营/廉政")
    description: str = Field(..., description="风险事件描述")
    detected_in: Optional[str] = Field(default=None, description="发现于（审计项目 ID）")
    related_regulations: List[str] = Field(default_factory=list, description="相关法规 ID 列表")
    risk_level: RiskLevel = Field(..., description="风险等级")
    occurrence_time: Optional[datetime] = Field(default=None, description="发生时间")
    amount_involved: Optional[float] = Field(default=None, description="涉及金额")
    status: str = Field(default="open", description="状态：open/investigating/closed")
    attributes: Optional[dict] = Field(default=None, description="扩展属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "risk_001",
                "event_type": "财务",
                "description": "发现大额资金往来未履行审批程序",
                "detected_in": "audit_project_001",
                "related_regulations": ["reg_001"],
                "risk_level": "high",
                "occurrence_time": "2023-03-10T00:00:00",
                "amount_involved": 10000000.0,
                "status": "investigating"
            }
        }


# ==================== RAG Request/Response Models ====================
class RAGQueryRequest(BaseModel):
    """
    RAG 查询请求模型
    """
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    vector_top_k: int = Field(default=5, ge=1, le=20, description="向量检索 TopK")
    graph_hops: int = Field(default=2, ge=1, le=5, description="图谱跳跃层数")
    include_cases: bool = Field(default=True, description="是否包含关联案例")
    include_regulations: bool = Field(default=True, description="是否包含相关法规")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "政府投资建设项目审计的主要法律依据是什么？",
                "vector_top_k": 5,
                "graph_hops": 2,
                "include_cases": True,
                "include_regulations": True
            }
        }


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
    """
    RAG 查询响应模型
    强制输出格式，符合审计合规要求
    """
    answer: str = Field(..., description="核心结论")
    basis_clauses: List[BasisClause] = Field(default_factory=list, description="依据条款列表")
    related_cases: List[RelatedCase] = Field(default_factory=list, description="关联案例列表")
    confidence_score: float = Field(..., description="置信度", ge=0.0, le=1.0)
    trace_paths: List[TracePath] = Field(default_factory=list, description="溯源路径列表")
    validation_flags: ValidationFlags = Field(..., description="校验标志")
    risk_level: RiskLevel = Field(..., description="风险等级")
    compliance_suggestions: List[str] = Field(default_factory=list, description="合规建议列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "政府投资建设项目审计的主要法律依据包括《中华人民共和国审计法》...",
                "basis_clauses": [
                    {
                        "clause_id": "第二十二条",
                        "clause_content": "审计机关对政府投资和以政府投资为主的建设项目...",
                        "source": "中华人民共和国审计法",
                        "effectiveness_level": "法律"
                    }
                ],
                "related_cases": [
                    {
                        "case_id": "case_001",
                        "case_summary": "XX 公司虚增利润案",
                        "similarity_score": 0.85,
                        "outcome": "责令改正，罚款 500 万元"
                    }
                ],
                "confidence_score": 0.85,
                "trace_paths": [
                    {
                        "path_type": "graph",
                        "path_description": "从'政府投资'节点经 2 跳关联到'审计法'",
                        "nodes": ["政府投资", "建设项目", "审计法"]
                    }
                ],
                "validation_flags": {
                    "amount_validated": True,
                    "time_validated": True,
                    "uncertainty_notes": []
                },
                "risk_level": "low",
                "compliance_suggestions": ["建议重点关注招投标环节的合规性", "注意工程变更的审批程序"]
            }
        }


# ==================== Graph Schema Models ====================
class KGNode(BaseModel):
    """知识图谱节点模型"""
    label: str = Field(..., description="节点 Label")
    properties: dict = Field(..., description="节点属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "Organization",
                "properties": {
                    "name": "XX 公司",
                    "id": "org_001"
                }
            }
        }


class KGRelation(BaseModel):
    """知识图谱关系模型"""
    source: str = Field(..., description="源节点 ID")
    target: str = Field(..., description="目标节点 ID")
    type: str = Field(..., description="关系类型")
    properties: Optional[dict] = Field(default=None, description="关系属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "org_001",
                "target": "case_001",
                "type": "INVOLVED_IN",
                "properties": {
                    "role": "被审计单位"
                }
            }
        }


class TripleData(BaseModel):
    """三元组数据模型（用于批量导入）"""
    nodes: List[KGNode] = Field(..., description="节点列表")
    relations: List[KGRelation] = Field(..., description="关系列表")
