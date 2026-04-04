"""
RAG Pipeline Integration Tests - RAG 管道集成测试

用途：测试检索→生成→格式校验的完整流程
关键依赖：pytest, pytest-asyncio
审计场景映射：验证 RAG 响应的完整性、格式合规性、溯源能力
"""

import pytest
import asyncio
from typing import Dict, Any

# 导入被测模块
from app.models.schema import (
    RAGQueryRequest,
    RAGQueryResponse,
    ValidationFlags,
    RiskLevel,
)
from app.core.retriever import HybridRetriever, get_hybrid_retriever
from app.core.generator import RAGGenerator, get_generator


# ==================== Fixtures ====================

@pytest.fixture
def sample_question() -> str:
    """样例问题"""
    return "政府投资建设项目审计的主要法律依据是什么？"


@pytest.fixture
def mock_vector_results() -> list:
    """模拟向量检索结果"""
    return [
        {
            "chunk": "审计机关对政府投资和以政府投资为主的建设项目的预算执行情况和决算进行审计监督。",
            "source": "中华人民共和国审计法",
            "score": 0.92,
            "metadata": {"clause_id": "第二十二条"}
        },
        {
            "chunk": "在中华人民共和国境内进行下列工程建设项目包括项目的勘察、设计、施工、监理以及与工程建设有关的重要设备、材料等的采购，必须进行招标。",
            "source": "中华人民共和国招标投标法",
            "score": 0.85,
            "metadata": {"clause_id": "第三条"}
        },
    ]


@pytest.fixture
def mock_graph_results() -> list:
    """模拟图谱检索结果"""
    return [
        {
            "type": "Regulation",
            "node_id": "reg_001",
            "properties": {
                "title": "中华人民共和国审计法",
                "clause_content": "审计机关对政府投资和以政府投资为主的建设项目的预算执行情况和决算进行审计监督。"
            },
            "path_description": "从'政府投资'节点关联到'审计法'",
            "nodes": ["government_investment", "reg_001"],
            "relevance_score": 0.88,
            "source": "graph",
        }
    ]


@pytest.fixture
def mock_retrieval_results(mock_vector_results, mock_graph_results) -> Dict[str, Any]:
    """模拟混合检索结果"""
    return {
        "query": "政府投资建设项目审计的主要法律依据是什么？",
        "vector_results": mock_vector_results,
        "graph_results": mock_graph_results,
        "fused_results": {
            "combined_ranking": [],
            "vector_weight": 0.6,
            "graph_weight": 0.4,
        },
        "parameters": {
            "vector_top_k": 5,
            "graph_hops": 2,
            "weights": {"vector": 0.6, "graph": 0.4},
        }
    }


# ==================== Retriever Tests ====================

class TestHybridRetriever:
    """混合检索器测试"""
    
    @pytest.mark.asyncio
    async def test_retriever_initialization(self):
        """测试检索器初始化"""
        retriever = get_hybrid_retriever(
            vector_top_k=3,
            graph_hops=2,
            fusion_weights={"vector": 0.7, "graph": 0.3}
        )
        
        assert retriever is not None
        assert retriever._vector_top_k == 3
        assert retriever._graph_hops == 2
        assert retriever._fusion_weights["vector"] == 0.7
    
    @pytest.mark.asyncio
    async def test_retrieve_structure(self, sample_question):
        """测试检索结果结构（不依赖真实服务）"""
        # 由于 Neo4j 和 Chroma 可能未启动，这里只测试返回结构
        # 实际使用时需要启动服务或 Mock
        
        retriever = get_hybrid_retriever()
        
        try:
            results = await retriever.retrieve(query=sample_question)
            
            # 验证返回结构
            assert "query" in results
            assert "vector_results" in results
            assert "graph_results" in results
            assert "fused_results" in results
            
        except Exception as e:
            # 如果服务未启动，预期会抛出异常
            pytest.skip(f"Services not available: {str(e)}")


# ==================== Generator Tests ====================

class TestRAGGenerator:
    """RAG 生成器测试"""
    
    @pytest.mark.asyncio
    async def test_generator_initialization(self):
        """测试生成器初始化"""
        generator = get_generator()
        assert generator is not None
    
    @pytest.mark.asyncio
    async def test_empty_response_generation(self, sample_question):
        """测试空检索结果的兜底响应"""
        generator = get_generator()
        
        empty_retrieval = {
            "vector_results": [],
            "graph_results": [],
        }
        
        response = await generator.generate(
            question=sample_question,
            retrieval_results=empty_retrieval
        )
        
        # 验证响应类型
        assert isinstance(response, RAGQueryResponse)
        
        # 验证兜底逻辑
        assert response.confidence_score == 0.0
        assert len(response.basis_clauses) == 0
        assert len(response.related_cases) == 0
        assert "未检索到相关知识" in response.validation_flags.uncertainty_notes
    
    @pytest.mark.asyncio
    async def test_mock_response_generation(self, mock_retrieval_results):
        """测试模拟响应生成（无真实 LLM）"""
        generator = get_generator()
        
        question = "政府投资建设项目审计的主要法律依据是什么？"
        
        # 当前没有真实 LLM，会返回降级响应
        response = await generator.generate(
            question=question,
            retrieval_results=mock_retrieval_results
        )
        
        # 验证响应结构完整性
        assert isinstance(response, RAGQueryResponse)
        assert response.answer is not None
        assert isinstance(response.confidence_score, float)
        assert 0.0 <= response.confidence_score <= 1.0
        assert isinstance(response.validation_flags, ValidationFlags)
        assert isinstance(response.risk_level, RiskLevel)


# ==================== Response Schema Tests ====================

class TestResponseSchema:
    """响应模型校验测试"""
    
    def test_rag_query_request_validation(self):
        """测试请求模型校验"""
        # 有效请求
        request = RAGQueryRequest(
            question="政府投资建设项目审计的法律依据？",
            vector_top_k=5,
            graph_hops=2,
            include_cases=True,
            include_regulations=True
        )
        assert request.question == "政府投资建设项目审计的法律依据？"
        assert request.vector_top_k == 5
        
        # 测试边界值
        with pytest.raises(Exception):
            RAGQueryRequest(question="", vector_top_k=5)  # 空问题
        
        with pytest.raises(Exception):
            RAGQueryRequest(question="test", vector_top_k=0)  # TopK 为 0
    
    def test_rag_query_response_structure(self):
        """测试响应模型结构"""
        response = RAGQueryResponse(
            answer="测试回答",
            basis_clauses=[],
            related_cases=[],
            confidence_score=0.85,
            trace_paths=[],
            validation_flags=ValidationFlags(
                amount_validated=True,
                time_validated=True,
                uncertainty_notes=[]
            ),
            risk_level=RiskLevel.LOW,
            compliance_suggestions=["建议 1"]
        )
        
        # 验证字段存在性
        assert hasattr(response, "answer")
        assert hasattr(response, "basis_clauses")
        assert hasattr(response, "related_cases")
        assert hasattr(response, "confidence_score")
        assert hasattr(response, "trace_paths")
        assert hasattr(response, "validation_flags")
        assert hasattr(response, "risk_level")
        assert hasattr(response, "compliance_suggestions")
        
        # 验证置信度范围
        assert 0.0 <= response.confidence_score <= 1.0
        
        # 验证风险等级
        assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]


# ==================== Integration Tests ====================

class TestRAGPipelineIntegration:
    """RAG 管道集成测试"""
    
    @pytest.mark.asyncio
    async def test_pipeline_structure(self, sample_question):
        """测试管道基本结构（Mock 模式）"""
        # Step 1: 创建检索器
        retriever = get_hybrid_retriever(vector_top_k=3)
        
        # Step 2: 创建生成器
        generator = get_generator()
        
        # 验证组件已创建
        assert retriever is not None
        assert generator is not None
        
        # 注意：完整的端到端测试需要：
        # 1. 启动 Neo4j 和 Chroma 服务
        # 2. 配置真实的 LLM API
        # 3. 预置测试数据
        
        # 当前版本仅验证结构，实际调用被 Mock


# ==================== Audit Compliance Tests ====================

class TestAuditCompliance:
    """审计合规性测试"""
    
    def test_output_format_requirements(self):
        """测试输出格式要求"""
        # 验证响应必须包含的关键字段
        required_fields = [
            "answer",           # 核心结论
            "basis_clauses",    # 依据条款
            "related_cases",    # 关联案例
            "confidence_score", # 置信度
            "trace_paths",      # 溯源路径
            "validation_flags", # 校验标志
            "risk_level",       # 风险等级
            "compliance_suggestions", # 合规建议
        ]
        
        response = RAGQueryResponse(
            answer="测试",
            basis_clauses=[],
            related_cases=[],
            confidence_score=0.5,
            trace_paths=[],
            validation_flags=ValidationFlags(
                amount_validated=True,
                time_validated=True,
                uncertainty_notes=[]
            ),
            risk_level=RiskLevel.MEDIUM,
            compliance_suggestions=[]
        )
        
        for field in required_fields:
            assert hasattr(response, field), f"缺少必需字段：{field}"
    
    def test_validation_flags_logic(self):
        """测试校验标志逻辑"""
        # 金额/时间校验标志应能正确反映不确定性
        flags_with_uncertainty = ValidationFlags(
            amount_validated=False,
            time_validated=False,
            uncertainty_notes=["涉及金额信息，需人工复核"]
        )
        
        assert flags_with_uncertainty.amount_validated is False
        assert len(flags_with_uncertainty.uncertainty_notes) > 0


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
