"""
Generator Module - RAG 生成器

用途：基于检索结果生成审计合规回答，注入审计专用 Prompt 约束
关键依赖：langchain, LLM
审计场景映射：强制输出【依据条款】【关联案例】【置信度】【溯源路径】，禁止幻觉
健壮性：JSON 解析容错、空结果兜底、金额/时间二次校验提示
"""

from typing import Optional, Dict, Any, List
import json
import re
from datetime import datetime
from app.utils.logger import get_logger
from app.utils.prompts import audit_rag_prompt, format_audit_context
from app.config import settings
from app.models.schema import (
    RAGQueryResponse,
    BasisClause,
    RelatedCase,
    TracePath,
    ValidationFlags,
    RiskLevel,
)

logger = get_logger("generator")


class RAGGenerator:
    """
    RAG 生成器
    基于混合检索结果，使用 LLM 生成符合审计合规要求的回答
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        初始化生成器
        
        Args:
            llm_client: LLM 客户端实例（LangChain LLM 或兼容接口）
        """
        self._llm_client = llm_client
        logger.info("rag_generator_initialized")
    
    def set_llm_client(self, llm_client: Any):
        """
        注入 LLM 客户端
        
        Args:
            llm_client: LLM 客户端实例
        """
        self._llm_client = llm_client
        logger.info("llm_client_injected")
    
    async def generate(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        生成 RAG 回答
        
        Args:
            question: 用户问题
            retrieval_results: 混合检索结果
            
        Returns:
            RAGQueryResponse 响应对象
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized. Call set_llm_client() first.")
        
        # 处理空检索结果的兜底策略
        vector_results = retrieval_results.get("vector_results", [])
        graph_results = retrieval_results.get("graph_results", [])
        
        if not vector_results and not graph_results:
            logger.warning("empty_retrieval_results", question=question[:50])
            return self._generate_empty_response(question)
        
        # 构建 Prompt 上下文
        prompt_inputs = format_audit_context(
            question=question,
            vector_results=vector_results,
            graph_results=graph_results,
            vector_top_k=len(vector_results),
            graph_hops=retrieval_results.get("parameters", {}).get("graph_hops", 2),
            vector_weight=retrieval_results.get("parameters", {}).get("weights", {}).get("vector", 0.6),
            graph_weight=retrieval_results.get("parameters", {}).get("weights", {}).get("graph", 0.4),
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        try:
            # 调用 LLM
            response = await self._invoke_llm(prompt_inputs)
            
            # 解析响应
            parsed_response = self._parse_llm_response(response, question, retrieval_results)
            
            # 二次校验（金额/时间）
            validated_response = self._validate_amount_and_time(parsed_response, question)
            
            logger.info(
                "generation_completed",
                question=question[:50] + "..." if len(question) > 50 else question,
                confidence=validated_response.confidence_score,
                basis_count=len(validated_response.basis_clauses),
                case_count=len(validated_response.related_cases),
            )
            
            return validated_response
            
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            # 返回降级响应
            return self._generate_error_response(question, str(e))
    
    async def _invoke_llm(self, prompt_inputs: Dict[str, Any]) -> Any:
        """
        调用 LLM
        
        Args:
            prompt_inputs: Prompt 输入字典
            
        Returns:
            LLM 原始响应
        """
        # 格式化 Prompt
        formatted_prompt = audit_rag_prompt.format(**prompt_inputs)
        
        # 调用 LLM
        if hasattr(self._llm_client, "invoke"):
            response = await self._llm_client.invoke(formatted_prompt)
        else:
            response = await self._llm_client(formatted_prompt)
        
        return response
    
    def _parse_llm_response(
        self,
        response: Any,
        question: str,
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        解析 LLM 响应为结构化对象
        
        Args:
            response: LLM 原始响应
            question: 原始问题
            retrieval_results: 检索结果
            
        Returns:
            RAGQueryResponse 对象
        """
        # 提取响应内容
        content = response
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict) and "text" in response:
            content = response["text"]
        
        content_str = str(content).strip()
        
        # 尝试提取 JSON
        json_content = self._extract_json_from_response(content_str)
        
        if json_content:
            try:
                data = json.loads(json_content)
                return self._build_response_from_dict(data, retrieval_results)
            except json.JSONDecodeError as e:
                logger.warning("json_parse_failed", error=str(e))
        
        # JSON 解析失败时，尝试从文本中提取关键信息
        return self._build_response_from_text(content_str, question, retrieval_results)
    
    def _extract_json_from_response(self, text: str) -> Optional[str]:
        """
        从响应文本中提取 JSON 内容
        
        Args:
            text: 响应文本
            
        Returns:
            JSON 字符串或 None
        """
        # 处理 markdown 代码块
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1)
        
        if "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1)
        
        # 尝试直接解析整个文本
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        
        return None
    
    def _build_response_from_dict(
        self,
        data: Dict[str, Any],
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        从字典构建响应对象
        
        Args:
            data: 解析后的字典
            retrieval_results: 检索结果
            
        Returns:
            RAGQueryResponse 对象
        """
        # 构建依据条款
        basis_clauses = []
        for clause in data.get("basis_clauses", []):
            basis_clauses.append(BasisClause(
                clause_id=clause.get("clause_id", ""),
                clause_content=clause.get("clause_content", ""),
                source=clause.get("source", ""),
                effectiveness_level=clause.get("effectiveness_level", ""),
            ))
        
        # 构建关联案例
        related_cases = []
        for case in data.get("related_cases", []):
            related_cases.append(RelatedCase(
                case_id=case.get("case_id", ""),
                case_summary=case.get("case_summary", ""),
                similarity_score=case.get("similarity_score", 0.0),
                outcome=case.get("outcome"),
            ))
        
        # 构建溯源路径
        trace_paths = []
        for path in data.get("trace_paths", []):
            trace_paths.append(TracePath(
                path_type=path.get("path_type", "unknown"),
                path_description=path.get("path_description", ""),
                nodes=path.get("nodes", []),
            ))
        
        # 如果响应中没有提供溯源路径，从检索结果中构建
        if not trace_paths:
            trace_paths = self._build_trace_paths_from_retrieval(retrieval_results)
        
        # 构建校验标志
        validation_flags_data = data.get("validation_flags", {})
        validation_flags = ValidationFlags(
            amount_validated=validation_flags_data.get("amount_validated", True),
            time_validated=validation_flags_data.get("time_validated", True),
            uncertainty_notes=validation_flags_data.get("uncertainty_notes", []),
        )
        
        return RAGQueryResponse(
            answer=data.get("answer", ""),
            basis_clauses=basis_clauses,
            related_cases=related_cases,
            confidence_score=data.get("confidence_score", 0.5),
            trace_paths=trace_paths,
            validation_flags=validation_flags,
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            compliance_suggestions=data.get("compliance_suggestions", []),
        )
    
    def _build_response_from_text(
        self,
        text: str,
        question: str,
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        从纯文本构建响应（降级方案）
        
        Args:
            text: 响应文本
            question: 原始问题
            retrieval_results: 检索结果
            
        Returns:
            RAGQueryResponse 对象
        """
        # 从文本中提取可能的答案
        answer = text[:500] if len(text) > 500 else text
        
        # 从检索结果构建溯源路径
        trace_paths = self._build_trace_paths_from_retrieval(retrieval_results)
        
        return RAGQueryResponse(
            answer=answer,
            basis_clauses=[],
            related_cases=[],
            confidence_score=0.3,  # 降低置信度
            trace_paths=trace_paths,
            validation_flags=ValidationFlags(
                amount_validated=False,
                time_validated=False,
                uncertainty_notes=["响应格式解析失败，建议人工复核"],
            ),
            risk_level=RiskLevel.MEDIUM,
            compliance_suggestions=["建议查阅相关法规原文进行确认"],
        )
    
    def _build_trace_paths_from_retrieval(
        self,
        retrieval_results: Dict[str, Any],
    ) -> List[TracePath]:
        """
        从检索结果构建溯源路径
        
        Args:
            retrieval_results: 检索结果
            
        Returns:
            溯源路径列表
        """
        trace_paths = []
        
        # 向量检索路径
        vector_results = retrieval_results.get("vector_results", [])
        if vector_results:
            sources = list(set(r.get("source", "unknown") for r in vector_results[:3]))
            trace_paths.append(TracePath(
                path_type="vector",
                path_description=f"通过向量相似度检索到 {len(vector_results)} 个相关文档片段",
                nodes=sources,
            ))
        
        # 图谱检索路径
        graph_results = retrieval_results.get("graph_results", [])
        if graph_results:
            paths = [r.get("path_description", "") for r in graph_results[:3]]
            nodes = []
            for r in graph_results[:3]:
                nodes.extend(r.get("nodes", []))
            trace_paths.append(TracePath(
                path_type="graph",
                path_description="; ".join(paths),
                nodes=list(set(nodes)),
            ))
        
        return trace_paths
    
    def _validate_amount_and_time(
        self,
        response: RAGQueryResponse,
        question: str,
    ) -> RAGQueryResponse:
        """
        二次校验金额和时间信息
        
        Args:
            response: RAG 响应
            question: 原始问题
            
        Returns:
            校验后的响应
        """
        uncertainty_notes = list(response.validation_flags.uncertainty_notes)
        
        # 检查问题中是否包含金额
        amount_pattern = r'\d+[万万亿千亿千百万]?(元|万元|亿元)?'
        if re.search(amount_pattern, question):
            # 问题涉及金额，需要校验
            if not response.basis_clauses:
                uncertainty_notes.append("涉及金额信息，但未找到明确的法规依据，建议人工复核")
                response.validation_flags.amount_validated = False
        
        # 检查问题中是否包含时间
        time_patterns = [r'\d{4}年', r'\d{4}-\d{2}', r'近期', '当前', '目前']
        for pattern in time_patterns:
            if re.search(pattern, question):
                if not response.validation_flags.time_validated:
                    uncertainty_notes.append("涉及时间信息，请注意时效性验证")
                break
        
        response.validation_flags.uncertainty_notes = uncertainty_notes
        return response
    
    def _generate_empty_response(self, question: str) -> RAGQueryResponse:
        """
        生成空检索结果的兜底响应
        
        Args:
            question: 原始问题
            
        Returns:
            兜底响应
        """
        return RAGQueryResponse(
            answer="抱歉，当前知识库中未检索到与您问题相关的信息。建议您：\n"
                   "1. 尝试使用不同的关键词重新查询\n"
                   "2. 联系审计专业人员获取帮助\n"
                   "3. 查阅相关法规原文",
            basis_clauses=[],
            related_cases=[],
            confidence_score=0.0,
            trace_paths=[],
            validation_flags=ValidationFlags(
                amount_validated=True,
                time_validated=True,
                uncertainty_notes=["未检索到相关知识"],
            ),
            risk_level=RiskLevel.LOW,
            compliance_suggestions=[],
        )
    
    def _generate_error_response(
        self,
        question: str,
        error_message: str,
    ) -> RAGQueryResponse:
        """
        生成错误响应
        
        Args:
            question: 原始问题
            error_message: 错误信息
            
        Returns:
            错误响应
        """
        return RAGQueryResponse(
            answer=f"生成回答时遇到技术问题：{error_message}。请稍后重试或联系技术支持。",
            basis_clauses=[],
            related_cases=[],
            confidence_score=0.0,
            trace_paths=[],
            validation_flags=ValidationFlags(
                amount_validated=False,
                time_validated=False,
                uncertainty_notes=[f"生成失败：{error_message}"],
            ),
            risk_level=RiskLevel.HIGH,
            compliance_suggestions=[],
        )


def get_generator(llm_client: Optional[Any] = None) -> RAGGenerator:
    """
    工厂函数：创建生成器实例
    
    Args:
        llm_client: LLM 客户端（可选）
        
    Returns:
        RAGGenerator 实例
    """
    return RAGGenerator(llm_client)
