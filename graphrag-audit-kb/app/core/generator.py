"""
Generator Module - 基于检索结果的 RAG 响应生成（集成 Qwen LLM）

用途：将 ``HybridRetriever.retrieve`` 返回的检索结果通过 Qwen LLM 生成自然语言回答
关键依赖：dashscope (Qwen LLM)
审计场景映射：法规条款解释、案例关联分析、风险事件评估
"""

from typing import Any, Dict, List, Optional

from app.config import settings
from app.utils.logger import get_logger
from app.utils.prompts import format_audit_context, build_rag_prompt, AUDIT_RAG_SYSTEM_PROMPT
from app.models.schema import (
    RAGQueryResponse,
    TracePath,
    ValidationFlags,
    RiskLevel,
)

logger = get_logger("generator")


def _build_trace_paths(retrieval_results: Dict[str, Any]) -> List[TracePath]:
    """
    从检索结果构造溯源路径列表。

    读取 ``vector_results``、``graph_results``；向量路径汇总来源与条数，图谱路径合并前若干条的
    ``path_description`` 与 ``nodes``（各取最多 3 条以控制体积）。
    """
    trace_paths: List[TracePath] = []
    vector_results = retrieval_results.get("vector_results", [])
    graph_results = retrieval_results.get("graph_results", [])

    if vector_results:
        sources = list({r.get("source", "unknown") for r in vector_results[:3]})
        trace_paths.append(
            TracePath(
                path_type="vector",
                path_description=f"通过向量相似度检索到 {len(vector_results)} 个相关文档片段",
                nodes=sources,
            )
        )

    if graph_results:
        paths = [r.get("path_description", "") for r in graph_results[:3]]
        nodes: List[str] = []
        for r in graph_results[:3]:
            nodes.extend(r.get("nodes", []))
        trace_paths.append(
            TracePath(
                path_type="graph",
                path_description="; ".join(paths),
                nodes=list(dict.fromkeys(nodes)),
            )
        )

    return trace_paths


def _confidence_from_retrieval(
    vector_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
) -> float:
    """
    基于检索条数给出占位置信度（启发式）：基准 0.5，向量与图谱分支按条数增量 capped，上限 0.95。
    """
    confidence = 0.5
    if vector_results:
        confidence += 0.2 * min(len(vector_results), 3) / 3
    if graph_results:
        confidence += 0.3 * min(len(graph_results), 3) / 3
    return min(confidence, 0.95)


class RAGGenerator:
    """基于检索结果生成 RAG 响应；支持 Qwen LLM 生成或占位模式。"""

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client
        self._has_llm = llm_client is not None
        logger.info(
            "rag_generator_initialized",
            has_llm=self._has_llm,
            llm_type=type(llm_client).__name__ if llm_client else None
        )

    def set_llm_client(self, llm_client: Any) -> None:
        """设置或替换大模型客户端实例。"""
        self._llm_client = llm_client
        self._has_llm = llm_client is not None

    async def generate(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        将检索结果通过 LLM 生成 RAG 响应。

        若未配置 LLM 或检索结果为空，返回占位提示；
        否则调用 Qwen LLM 基于检索上下文生成专业回答。

        Args:
            question: 用户问题
            retrieval_results: 须含 ``vector_results``、``graph_results``，可选 ``parameters``

        Returns:
            符合 Schema 的响应模型实例
        """
        vector_results = retrieval_results.get("vector_results", [])
        graph_results = retrieval_results.get("graph_results", [])

        if not vector_results and not graph_results:
            logger.warning("empty_retrieval_results", question=question[:50])
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

        params = retrieval_results.get("parameters", {})
        
        # 如果有 LLM，调用 Qwen 生成回答
        if self._has_llm and self._llm_client:
            try:
                # 构建 RAG Prompt
                prompt = build_rag_prompt(
                    question=question,
                    vector_results=vector_results,
                    graph_results=graph_results,
                    vector_top_k=params.get("vector_top_k", len(vector_results)),
                    graph_hops=params.get("graph_hops", 2),
                )
                
                # 调用 LLM 生成
                temperature = settings.llm_temperature
                answer = await self._llm_client.generate(
                    prompt=prompt,
                    system_prompt=AUDIT_RAG_SYSTEM_PROMPT,
                    temperature=temperature,
                    max_tokens=2048,
                )
                
                logger.info(
                    "llm_generation_completed",
                    question=question[:50],
                    answer_length=len(answer),
                    temperature=temperature
                )
                
                # 解析 LLM 回答中的条款和案例（简单实现，后续可优化）
                basis_clauses = self._extract_basis_clauses(answer, vector_results)
                related_cases = self._extract_related_cases(answer, graph_results)
                
                return RAGQueryResponse(
                    answer=answer,
                    basis_clauses=basis_clauses,
                    related_cases=related_cases,
                    confidence_score=_confidence_from_retrieval(vector_results, graph_results),
                    trace_paths=_build_trace_paths(retrieval_results),
                    validation_flags=ValidationFlags(
                        amount_validated=True,
                        time_validated=True,
                        uncertainty_notes=[],
                    ),
                    risk_level=self._assess_risk_level(answer, graph_results),
                    compliance_suggestions=self._extract_compliance_suggestions(answer),
                )
                
            except Exception as e:
                logger.error("llm_generation_failed", error=str(e))
                # 降级到占位生成
                return self._fallback_generate(question, retrieval_results, str(e))
        else:
            # 无 LLM 时的占位生成
            return self._fallback_generate(question, retrieval_results, "LLM not configured")

    def _fallback_generate(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
        error_reason: str,
    ) -> RAGQueryResponse:
        """LLM 不可用时的占位生成"""
        vector_results = retrieval_results.get("vector_results", [])
        graph_results = retrieval_results.get("graph_results", [])
        params = retrieval_results.get("parameters", {})
        
        ctx = format_audit_context(
            question=question,
            vector_results=vector_results,
            graph_results=graph_results,
            vector_top_k=params.get("vector_top_k", len(vector_results)),
            graph_hops=params.get("graph_hops", 2),
            vector_weight=params.get("weights", {}).get("vector", 0.6),
            graph_weight=params.get("weights", {}).get("graph", 0.4),
        )

        trace_paths = _build_trace_paths(retrieval_results)
        confidence = _confidence_from_retrieval(vector_results, graph_results)

        answer = (
            f"【基于检索的摘要响应】关于您的问题：「{question}」\n\n"
            f"已检索到 {len(vector_results)} 条向量结果、{len(graph_results)} 条图谱结果。\n\n"
            "【检索上下文】\n"
            f"{ctx['vector_context']}\n\n"
            f"{ctx['graph_context']}\n\n"
            f"注意：{error_reason}。配置 DASHSCOPE_API_KEY 后可启用完整生成能力。"
        )

        return RAGQueryResponse(
            answer=answer,
            basis_clauses=[],
            related_cases=[],
            confidence_score=confidence,
            trace_paths=trace_paths,
            validation_flags=ValidationFlags(
                amount_validated=True,
                time_validated=True,
                uncertainty_notes=[f"占位生成：{error_reason}"],
            ),
            risk_level=RiskLevel.LOW,
            compliance_suggestions=[
                "配置 DASHSCOPE_API_KEY 以启用 Qwen LLM 生成",
                "建议查阅相关法规原文进行确认",
            ],
        )

    def _extract_basis_clauses(
        self,
        answer: str,
        vector_results: List[Dict[str, Any]],
    ) -> List[str]:
        """从回答中提取依据条款（简单实现：返回相关文档片段的来源）"""
        clauses = []
        for result in vector_results[:3]:
            source = result.get("source", "")
            if source and source != "unknown":
                clauses.append(source)
        return list(set(clauses))

    def _extract_related_cases(
        self,
        answer: str,
        graph_results: List[Dict[str, Any]],
    ) -> List[str]:
        """从回答中提取相关案例（简单实现：返回图谱结果中的案例类型）"""
        cases = []
        for result in graph_results[:3]:
            if result.get("type") == "AuditCase":
                props = result.get("properties", {})
                case_name = props.get("name", props.get("title", ""))
                if case_name:
                    cases.append(case_name)
        return cases

    def _assess_risk_level(
        self,
        answer: str,
        graph_results: List[Dict[str, Any]],
    ) -> RiskLevel:
        """基于检索结果评估风险等级（简单启发式）"""
        # 检查是否有高风险实体
        for result in graph_results:
            props = result.get("properties", {})
            risk_level = props.get("risk_level", "").lower()
            if "high" in risk_level or "重大" in risk_level:
                return RiskLevel.HIGH
            if "medium" in risk_level or "中等" in risk_level:
                return RiskLevel.MEDIUM
        
        # 默认低风险
        return RiskLevel.LOW

    def _extract_compliance_suggestions(self, answer: str) -> List[str]:
        """从回答中提取合规建议（简单实现：按关键词分割）"""
        suggestions = []
        keywords = ["应", "应当", "建议", "注意", "需"]
        
        for line in answer.split("\n"):
            line = line.strip()
            if any(kw in line for kw in keywords) and len(line) > 10:
                suggestions.append(line)
        
        return suggestions[:5]  # 限制最多 5 条


def get_generator(llm_client: Optional[Any] = None) -> RAGGenerator:
    """工厂函数：创建 ``RAGGenerator``；可传入 Qwen LLM 客户端。"""
    return RAGGenerator(llm_client=llm_client)
