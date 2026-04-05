"""
Generator Module - 基于检索结果的 RAG 响应生成（无 LLM 最小闭环）

用途：将 ``HybridRetriever.retrieve`` 返回的检索结果组装为 ``RAGQueryResponse``；
可选注入 ``llm_client`` 供后续扩展，当前默认不调用大模型。
"""

from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger
from app.utils.prompts import format_audit_context
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
    """基于检索结果生成 RAG 响应；可选保留 ``llm_client`` 供后续扩展。"""

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client
        logger.info("rag_generator_initialized", has_llm=llm_client is not None)

    def set_llm_client(self, llm_client: Any) -> None:
        """设置或替换大模型客户端实例（当前 ``generate`` 未调用，仅预留）。"""
        self._llm_client = llm_client

    async def generate(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
    ) -> RAGQueryResponse:
        """
        将检索结果格式化为 ``RAGQueryResponse``。

        若向量与图谱结果均为空，返回固定提示与置信度 0、空溯源；
        否则拼接 ``format_audit_context`` 文本为 ``answer``，``basis_clauses`` 与 ``related_cases`` 当前保持空列表，
        ``validation_flags`` 中注明占位生成说明。

        Args:
            question: 用户问题
            retrieval_results: 须含 ``vector_results``、``graph_results``，可选 ``parameters``（与检索器返回一致）

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
            "当前未接入大语言模型；配置 LLM_API_KEY 后可替换为深度生成回答。"
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
                uncertainty_notes=["当前为基于检索的占位生成，未接入真实 LLM"],
            ),
            risk_level=RiskLevel.LOW,
            compliance_suggestions=[
                "配置 LLM API 密钥以启用完整生成能力",
                "建议查阅相关法规原文进行确认",
            ],
        )


def get_generator(llm_client: Optional[Any] = None) -> RAGGenerator:
    """工厂函数：创建 ``RAGGenerator``；``llm_client`` 默认为 ``None``（无 LLM 占位生成）。"""
    return RAGGenerator(llm_client=llm_client)
