"""
Knowledge Graph Schema - 图本体定义

用途：定义审计领域知识图谱的节点 Label、关系 Type 和属性约束
关键依赖：无（纯数据定义）
审计场景映射：支持审计案例、法规条款、风险事件、组织机构等核心实体及关系
"""

from typing import List, Dict, Set


# ==================== Node Labels ====================
# 审计领域核心实体类型
NODE_LABELS: Set[str] = {
    "Organization",      # 组织机构：被审计单位、监管机构、中介机构
    "Regulation",        # 法规条款：法律、行政法规、部门规章、地方规定、内部制度
    "AuditCase",         # 审计案例：历史审计项目、违规案例、处罚案例
    "RiskEvent",         # 风险事件：审计发现的风险点、异常事项
    "Person",            # 人员：法定代表人、负责人、经办人
    "Project",           # 项目：审计项目、建设项目
    "Amount",            # 金额：涉及金额、处罚金额
    "Time",              # 时间：发生时间、审计时间
}

# ==================== Relationship Types ====================
# 审计领域关系类型定义
RELATIONSHIP_TYPES: Set[str] = {
    # 组织相关关系
    "OWNED_BY",          # 归属于（下级单位属于上级单位）
    "SUPERVISED_BY",     # 受监管于（企业受监管机构监管）
    "AFFILIATED_WITH",   # 隶属于
    
    # 法规相关关系
    "ISSUED_BY",         # 由...发布（法规由机关发布）
    "APPLIES_TO",        # 适用于（法规适用于某类组织或事项）
    "AMENDED_BY",        # 被...修订
    "REPEALED_BY",       # 被...废止
    "REFERENCES",        # 引用（法规引用其他法规）
    
    # 案例相关关系
    "INVOLVED_IN",       # 卷入（组织卷入案例）
    "PENALIZED_BY",      # 被...处罚
    "DETECTED_IN",       # 在...中发现（风险事件在审计中发现）
    "SIMILAR_TO",        # 类似于（案例之间的相似关系）
    
    # 违规相关关系
    "VIOLATED",          # 违反（行为违反法规）
    "COMMITTED_BY",      # 由...实施
    "RESULTED_IN",       # 导致（违规行为导致后果）
    
    # 时间/金额相关关系
    "OCCURRED_AT",       # 发生于（事件发生时间）
    "AMOUNT_INVOLVED",   # 涉及金额
    "DURING_PERIOD",     # 在...期间
    
    # 项目相关关系
    "PART_OF_PROJECT",   # 属于某项目
    "AUDITED_IN",        # 在...中被审计
}

# ==================== Node Property Schemas ====================
# 各节点类型的属性定义
NODE_PROPERTY_SCHEMAS: Dict[str, Dict[str, str]] = {
    "Organization": {
        "id": "str",           # 唯一标识
        "name": "str",         # 组织名称（必填）
        "org_type": "str",     # 组织类型
        "industry": "str",     # 所属行业
        "registration_code": "str",  # 统一社会信用代码
        "region": "str",       # 所属地区
    },
    "Regulation": {
        "id": "str",           # 唯一标识
        "title": "str",        # 法规标题（必填）
        "clause_id": "str",    # 条款编号
        "clause_content": "str",  # 条款内容（必填）
        "effectiveness_level": "str",  # 效力等级
        "issuing_authority": "str",  # 发布机关
        "effective_date": "datetime",  # 生效日期
        "status": "str",       # 状态：effective/abolished/amended
    },
    "AuditCase": {
        "id": "str",           # 唯一标识
        "case_number": "str",  # 案例编号
        "title": "str",        # 案例标题（必填）
        "summary": "str",      # 案例摘要（必填）
        "violation_types": "list[str]",  # 违规类型列表
        "amount_involved": "float",  # 涉及金额
        "penalty_amount": "float",   # 处罚金额
        "outcome": "str",      # 处理结果
        "case_date": "datetime",  # 案例日期
    },
    "RiskEvent": {
        "id": "str",           # 唯一标识
        "event_type": "str",   # 风险类型（必填）
        "description": "str",  # 风险描述（必填）
        "risk_level": "str",   # 风险等级：low/medium/high
        "amount_involved": "float",  # 涉及金额
        "occurrence_time": "datetime",  # 发生时间
        "status": "str",       # 状态：open/investigating/closed
    },
    "Person": {
        "id": "str",           # 唯一标识
        "name": "str",         # 姓名（必填）
        "position": "str",     # 职务
        "organization_id": "str",  # 所属组织 ID
    },
    "Project": {
        "id": "str",           # 唯一标识
        "name": "str",         # 项目名称（必填）
        "project_type": "str", # 项目类型
        "total_amount": "float",  # 总投资金额
        "start_date": "datetime",  # 开始日期
        "end_date": "datetime",    # 结束日期
    },
}

# ==================== Relationship Property Schemas ====================
# 各关系类型的属性定义
RELATIONSHIP_PROPERTY_SCHEMAS: Dict[str, Dict[str, str]] = {
    "INVOLVED_IN": {
        "role": "str",         # 在案例中的角色：被审计单位/关联方/中介机构
        "involvement_degree": "str",  # 参与程度
    },
    "VIOLATED": {
        "violation_date": "datetime",  # 违规日期
        "severity": "str",     # 严重程度
    },
    "PENALIZED_BY": {
        "penalty_date": "datetime",  # 处罚日期
        "penalty_type": "str",  # 处罚类型
    },
    "OCCURRED_AT": {
        "precision": "str",    # 时间精度：exact/approximate
    },
    "AMOUNT_INVOLVED": {
        "currency": "str",     # 币种
        "amount_type": "str",  # 金额类型：涉及/处罚/追回
    },
}

# ==================== Index Definitions ====================
# Neo4j 索引定义（用于优化查询性能）。每项为 dict：
# - ``label``：节点标签；``properties``：需建 B-tree 索引的属性名列表（每个属性一条 CREATE INDEX 语句）
INDEX_DEFINITIONS: List[dict] = [
    {"label": "Organization", "properties": ["name", "registration_code"]},
    {"label": "Regulation", "properties": ["title", "clause_id"]},
    {"label": "AuditCase", "properties": ["case_number", "title"]},
    {"label": "RiskEvent", "properties": ["event_type", "risk_level"]},
    {"label": "Person", "properties": ["name"]},
    {"label": "Project", "properties": ["name", "project_type"]},
]

# ==================== Constraint Definitions ====================
# Neo4j 约束定义（用于保证数据完整性）。每项为 dict：
# - ``label``：节点标签；``property``：唯一约束属性名；``type``：当前仅使用 ``unique``
CONSTRAINT_DEFINITIONS: List[dict] = [
    {"label": "Organization", "property": "id", "type": "unique"},
    {"label": "Regulation", "property": "id", "type": "unique"},
    {"label": "AuditCase", "property": "id", "type": "unique"},
    {"label": "RiskEvent", "property": "id", "type": "unique"},
    {"label": "Person", "property": "id", "type": "unique"},
    {"label": "Project", "property": "id", "type": "unique"},
]


def get_node_labels() -> Set[str]:
    """获取所有节点 Label"""
    return NODE_LABELS.copy()


def get_relationship_types() -> Set[str]:
    """获取所有关系类型"""
    return RELATIONSHIP_TYPES.copy()


def get_node_schema(label: str) -> Dict[str, str]:
    """获取指定节点的属性 schema"""
    return NODE_PROPERTY_SCHEMAS.get(label, {})


def get_relationship_schema(rel_type: str) -> Dict[str, str]:
    """获取指定关系的属性 schema"""
    return RELATIONSHIP_PROPERTY_SCHEMAS.get(rel_type, {})


def generate_cypher_constraints() -> List[str]:
    """
    生成 Neo4j 5.x 风格唯一约束创建语句（``CREATE CONSTRAINT IF NOT EXISTS FOR (n:Label) ...``）。

    Returns:
        可逐条执行的 Cypher 字符串列表
    """
    statements = []
    for constraint in CONSTRAINT_DEFINITIONS:
        label = constraint["label"]
        property_name = constraint["property"]
        stmt = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        statements.append(stmt)
    return statements


def generate_cypher_indexes() -> List[str]:
    """
    生成 Neo4j 5.x 风格索引创建语句（``CREATE INDEX IF NOT EXISTS FOR (n:Label) ON (n.prop)``）。

    Returns:
        可逐条执行的 Cypher 字符串列表
    """
    statements = []
    for index in INDEX_DEFINITIONS:
        label = index["label"]
        for prop in index["properties"]:
            stmt = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
            statements.append(stmt)
    return statements
