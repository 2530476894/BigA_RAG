"""
Generator Module - 基于检索结果的 RAG 响应生成（集成 Qwen LLM）

用途：将 ``HybridRetriever.retrieve`` 返回的检索结果通过 Qwen LLM 生成自然语言回答
关键依赖：dashscope (Qwen LLM)
审计场景映射：法规条款解释、案例关联分析、风险事件评估
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.utils.logger import get_logger
from app.utils.prompts import format_audit_context, build_rag_prompt, AUDIT_RAG_SYSTEM_PROMPT
from app.models.schema import (
    RAGQueryResponse,
    TracePath,
    ValidationFlags,
    RiskLevel,
    RetrievalEvidenceItem,
    BasisClause,
    RelatedCase,
)

logger = get_logger("generator")

_CLAUSE_PREVIEW_LEN = 200


def _slice_vector_for_prompt(
    vector_results: List[Dict[str, Any]],
    vector_top_k: int,
) -> List[Dict[str, Any]]:
    return vector_results[:vector_top_k]


def _build_retrieval_evidence(
    sliced_vector_results: List[Dict[str, Any]],
) -> List[RetrievalEvidenceItem]:
    items: List[RetrievalEvidenceItem] = []
    for ref_index, result in enumerate(sliced_vector_results, start=1):
        meta = result.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        cid = result.get("chunk_id", "") or "unknown"
        items.append(
            RetrievalEvidenceItem(
                ref_index=ref_index,
                chunk_id=cid,
                text=result.get("chunk", "") or "",
                source=result.get("source", "unknown") or "unknown",
                score=float(result.get("score", 0.0)),
                metadata=dict(meta),
            )
        )
    return items


def _parse_answer_vector_refs(answer: str, max_valid_ref: int) -> List[int]:
    if max_valid_ref < 1:
        return []
    seen: Set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if 1 <= n <= max_valid_ref:
            seen.add(n)
    return sorted(seen)


def _build_trace_paths(retrieval_results: Dict[str, Any]) -> List[TracePath]:
    """
    从检索结果构造溯源路径列表。

    读取 ``vector_results``、``graph_results``、``entities``；包含实体识别、向量检索和图谱检索的溯源信息。
    """
    trace_paths: List[TracePath] = []
    vector_results = retrieval_results.get("vector_results", [])
    graph_results = retrieval_results.get("graph_results", [])
    entities = retrieval_results.get("entities", [])

    # 实体识别溯源
    if entities:
        entity_types = list({e.get("type", "unknown") for e in entities})
        trace_paths.append(
            TracePath(
                path_type="entity_extraction",
                path_description=f"通过LLM识别出 {len(entities)} 个审计领域实体：{', '.join(entity_types)}",
                nodes=[e.get("text", "") for e in entities[:5]],  # 限制显示数量
            )
        )

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
                retrieval_evidence=[],
                answer_cited_vector_refs=[],
                validation_flags=ValidationFlags(
                    amount_validated=True,
                    time_validated=True,
                    uncertainty_notes=["未检索到相关知识"],
                ),
                risk_level=RiskLevel.LOW,
                compliance_suggestions=[],
            )

        params = retrieval_results.get("parameters", {})
        vector_top_k = params.get("vector_top_k", len(vector_results))
        sliced = _slice_vector_for_prompt(vector_results, vector_top_k)
        retrieval_evidence = _build_retrieval_evidence(sliced)

        # 如果有 LLM，调用 Qwen 生成回答
        if self._has_llm and self._llm_client:
            try:
                # 构建 RAG Prompt
                prompt = build_rag_prompt(
                    question=question,
                    vector_results=vector_results,
                    graph_results=graph_results,
                    vector_top_k=vector_top_k,
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
                cited = _parse_answer_vector_refs(answer, len(sliced))
                
                return RAGQueryResponse(
                    answer=answer,
                    basis_clauses=basis_clauses,
                    related_cases=related_cases,
                    confidence_score=_confidence_from_retrieval(vector_results, graph_results),
                    trace_paths=_build_trace_paths(retrieval_results),
                    retrieval_evidence=retrieval_evidence,
                    answer_cited_vector_refs=cited,
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
                return self._fallback_generate(
                    question, retrieval_results, str(e),
                    retrieval_evidence=retrieval_evidence,
                    vector_top_k=vector_top_k,
                )
        else:
            # 无 LLM 时的占位生成
            return self._fallback_generate(
                question, retrieval_results, "LLM not configured",
                retrieval_evidence=retrieval_evidence,
                vector_top_k=vector_top_k,
            )

    def _fallback_generate(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
        error_reason: str,
        retrieval_evidence: List[RetrievalEvidenceItem],
        vector_top_k: int,
    ) -> RAGQueryResponse:
        """LLM 不可用时的占位生成"""
        vector_results = retrieval_results.get("vector_results", [])
        graph_results = retrieval_results.get("graph_results", [])
        params = retrieval_results.get("parameters", {})
        
        ctx = format_audit_context(
            question=question,
            vector_results=vector_results,
            graph_results=graph_results,
            vector_top_k=params.get("vector_top_k", vector_top_k),
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

        sliced = _slice_vector_for_prompt(vector_results, vector_top_k)
        cited = _parse_answer_vector_refs(answer, len(sliced))

        return RAGQueryResponse(
            answer=answer,
            basis_clauses=[],
            related_cases=[],
            confidence_score=confidence,
            trace_paths=trace_paths,
            retrieval_evidence=retrieval_evidence,
            answer_cited_vector_refs=cited,
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
    ) -> List[BasisClause]:
        """依据向量检索前 3 条构造条款摘要（元数据 + 正文截断）。"""
        seen: Set[Tuple[str, str]] = set()
        clauses: List[BasisClause] = []
        for result in vector_results[:3]:
            meta = result.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            clause_id = (meta.get("clause_id") or "").strip() or "未标注"
            source = (result.get("source") or "unknown").strip() or "unknown"
            key = (clause_id, source)
            if key in seen:
                continue
            seen.add(key)
            chunk = result.get("chunk", "") or ""
            preview = chunk[:_CLAUSE_PREVIEW_LEN] if chunk else ""
            eff = (meta.get("effectiveness_level") or "未标注").strip() or "未标注"
            clauses.append(
                BasisClause(
                    clause_id=clause_id,
                    clause_content=preview,
                    source=source,
                    effectiveness_level=eff,
                )
            )
        return clauses

    def _extract_related_cases(
        self,
        answer: str,
        graph_results: List[Dict[str, Any]],
    ) -> List[RelatedCase]:
        """从图谱结果中提取 AuditCase 条目（前 3 条相关）。"""
        cases: List[RelatedCase] = []
        for result in graph_results[:3]:
            if result.get("type") != "AuditCase":
                continue
            props = result.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            case_name = props.get("name") or props.get("title") or ""
            case_id = (props.get("case_id") or result.get("node_id") or "未标注").strip() or "未标注"
            summary = case_name.strip() if case_name else (str(props)[:_CLAUSE_PREVIEW_LEN] or "无摘要")
            raw_score = float(result.get("relevance_score", 0.0))
            similarity_score = max(0.0, min(1.0, raw_score))
            outcome = props.get("outcome")
            if outcome is not None:
                outcome = str(outcome)
            cases.append(
                RelatedCase(
                    case_id=case_id,
                    case_summary=summary,
                    similarity_score=similarity_score,
                    outcome=outcome,
                )
            )
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
