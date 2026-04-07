"""
Retriever Module - 混合检索器

用途：实现向量相似度 + Neo4j Cypher 多跳的混合检索，支持结果加权融合
关键依赖：neo4j, chromadb
审计场景映射：法规条款检索、案例关联查询、多跳关系发现
健壮性：空结果兜底、异常处理、日志追踪
"""

from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.utils.logger import get_logger
from app.services.neo4j_service import get_neo4j_service
from app.services.vector_service import get_vector_service
logger = get_logger("retriever")


class HybridRetriever:
    """
    混合检索器
    同时执行向量检索和图谱检索，并融合结果
    """
    
    def __init__(
        self,
        vector_top_k: int = None,
        graph_hops: int = None,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化混合检索器
        
        Args:
            vector_top_k: 向量检索 TopK 数量
            graph_hops: 图谱多跳层数
            fusion_weights: 融合权重配置 {"vector": 0.6, "graph": 0.4}
        """
        self._vector_top_k = vector_top_k or settings.vector_top_k
        self._graph_hops = graph_hops or settings.graph_hops
        self._fusion_weights = fusion_weights or settings.fusion_weights
        
        self._vector_service = get_vector_service()
        self._neo4j_service = get_neo4j_service()
        
        logger.info(
            "hybrid_retriever_initialized",
            vector_top_k=self._vector_top_k,
            graph_hops=self._graph_hops,
            weights=self._fusion_weights
        )
    
    async def retrieve(
        self,
        query: str,
        vector_top_k: Optional[int] = None,
        graph_hops: Optional[int] = None,
        include_cases: bool = True,
        include_regulations: bool = True,
    ) -> Dict[str, Any]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            vector_top_k: 向量检索 TopK（可选覆盖）
            graph_hops: 图谱跳跃层数（可选覆盖）
            include_cases: 是否包含案例
            include_regulations: 是否包含法规
            
        Returns:
            字典键包括：
            - ``query``：原始查询文本
            - ``vector_results``：向量检索结果列表
            - ``graph_results``：图谱检索结果列表
            - ``entities``：查询实体识别结果列表（供溯源路径等使用）
            - ``fused_results``：融合排序结果（当前 ``RAGGenerator`` 未使用，仅保留供扩展/分析）
            - ``parameters``：包含 ``vector_top_k``、``graph_hops``、``weights``（``vector``/``graph`` 权重）
        """
        top_k = vector_top_k or self._vector_top_k
        hops = graph_hops or self._graph_hops

        entities = await self._extract_entities(query)
        
        # 并发执行向量检索和图谱检索
        vector_results = await self._retrieve_vector(query, top_k)
        graph_results = await self._retrieve_graph(
            query, hops, include_cases, include_regulations, entities
        )
        
        # 融合结果（生成器当前仅使用 vector/graph 与 parameters，未读取 fused_results）
        fused_results = self._fuse_results(vector_results, graph_results)
        
        logger.info(
            "hybrid_retrieval_completed",
            query=query[:50] + "..." if len(query) > 50 else query,
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            fused_count=len(fused_results.get("combined_ranking", []))
        )
        
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "entities": entities,
            "fused_results": fused_results,
            "parameters": {
                "vector_top_k": top_k,
                "graph_hops": hops,
                "weights": self._fusion_weights,
            }
        }
    
    async def _retrieve_vector(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            向量检索结果列表
        """
        try:
            results = self._vector_service.similarity_search(query=query, k=top_k)
            return results
        except Exception as e:
            logger.error("vector_retrieval_failed", error=str(e))
            return []
    
    async def _retrieve_graph(
        self,
        query: str,
        hops: int,
        include_cases: bool,
        include_regulations: bool,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        知识图谱检索
        
        策略：
        1. 使用调用方传入的实体列表，搜索匹配的节点
        2. 从匹配节点出发进行多跳查询
        3. 根据 include 参数过滤结果类型
        
        Args:
            query: 查询文本
            hops: 跳跃层数
            include_cases: 是否包含案例
            include_regulations: 是否包含法规
            entities: 已由 ``retrieve`` 提取的实体列表
            
        Returns:
            图谱检索结果列表
        """
        try:
            graph_results = []
            
            # Step 2: 对每个实体进行搜索和多跳查询
            for entity in entities[:3]:  # 限制实体数量，避免过多查询
                entity_text = entity.get("text", "")
                entity_type = entity.get("type", "unknown")
                
                if not entity_text:
                    continue
                
                # 搜索匹配的节点
                matched_nodes = await self._search_matching_nodes(entity_text, entity_type)
                
                # 对每个匹配节点进行多跳查询
                for node in matched_nodes[:5]:  # 限制起始节点数量
                    node_label = node.get("_label", "Organization")
                    node_id = node.get("id", "")
                    
                    if not node_id:
                        continue
                    
                    # 执行多跳查询
                    multi_hop_results = self._neo4j_service.multi_hop_query(
                        start_label=node_label,
                        start_id=node_id,
                        hops=hops,
                    )
                    
                    # 格式化结果
                    for result in multi_hop_results:
                        formatted = self._format_graph_result(result, query)
                        if formatted:
                            graph_results.append(formatted)
            
            # Step 3: 去重和排序
            graph_results = self._deduplicate_graph_results(graph_results)
            
            # Step 4: 根据 include 参数过滤
            if not include_cases:
                graph_results = [r for r in graph_results if r.get("type") != "AuditCase"]
            if not include_regulations:
                graph_results = [r for r in graph_results if r.get("type") != "Regulation"]
            
            return graph_results
            
        except Exception as e:
            logger.error("graph_retrieval_failed", error=str(e))
            return []
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        从查询中提取审计领域实体
        
        使用LLM进行实体识别，如果失败则回退到关键词提取
        
        Args:
            query: 查询文本
            
        Returns:
            实体对象列表，每个包含type, text, confidence
        """
        from app.services.llm_entity_service import get_llm_entity_service
        
        entity_service = get_llm_entity_service()
        entities = await entity_service.extract_entities(query)
        
        if entities:
            return entities
        else:
            # 回退到关键词提取
            logger.warning("llm_entity_extraction_failed_fallback_to_keywords")
            keywords = self._extract_keywords_fallback(query)
            # 将关键词转换为实体格式
            return [
                {
                    "type": "unknown",
                    "text": keyword,
                    "confidence": 0.5
                }
                for keyword in keywords
            ]
    
    def _extract_keywords_fallback(self, query: str) -> List[str]:
        """
        回退的关键词提取方法（原_extract_keywords逻辑）
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        import re
        words = re.split(r'[\s,，.。;；:：!?！？]+', query)
        stopwords = {"的", "了", "是", "在", "和", "与", "及", "等", "什么", "如何", "怎么", "哪些"}
        keywords = [w for w in words if w and w not in stopwords and len(w) >= 2]
        return keywords[:10]  # 限制关键词数量
    
    async def _validate_entities_with_graph(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用图谱搜索验证和细化实体
        
        Args:
            entities: LLM提取的实体列表
            
        Returns:
            验证后的实体列表
        """
        validated_entities = []
        
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "")
            
            if not entity_text or entity_type == "unknown":
                continue
            
            # 使用图谱搜索验证实体
            matched_nodes = await self._search_matching_nodes(entity_text, entity_type)
            
            if matched_nodes:
                # 实体有效，保留
                validated_entity = entity.copy()
                validated_entity["graph_validation"] = True
                validated_entity["matched_nodes_count"] = len(matched_nodes)
                validated_entities.append(validated_entity)
            else:
                # 实体在图谱中未找到，降低置信度
                entity_copy = entity.copy()
                entity_copy["confidence"] *= 0.5
                entity_copy["graph_validation"] = False
                validated_entities.append(entity_copy)
        
        return validated_entities
    
    async def _search_matching_nodes(self, entity_text: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """
        搜索匹配实体的节点
        
        Args:
            entity_text: 实体文本
            entity_type: 实体类型（可选，用于优化搜索）
            
        Returns:
            匹配的节点列表
        """
        matched_nodes = []
        
        # 确定搜索的节点类型
        if entity_type and entity_type != "unknown":
            # 如果指定了实体类型，只搜索对应类型
            search_labels = [entity_type.capitalize()]
        else:
            # 否则搜索所有主要类型
            search_labels = ["Organization", "Regulation", "AuditCase", "RiskEvent"]
        
        # 使用语义搜索
        for label in search_labels:
            try:
                nodes = self._neo4j_service.semantic_search_nodes(
                    entity_text=entity_text,
                    entity_type=label.lower(),
                    similarity_threshold=settings.entity_linking.get("similarity_threshold", 0.7),
                    limit=settings.entity_linking.get("max_candidates", 5)
                )
                for node in nodes:
                    node["_label"] = label
                    matched_nodes.append(node)
            except Exception as e:
                logger.warning("semantic_node_search_failed", label=label, entity_text=entity_text, error=str(e))
        
        logger.info(
            "entity_node_search_completed",
            entity_text=entity_text,
            entity_type=entity_type,
            matched_count=len(matched_nodes)
        )
        
        return matched_nodes
    
    def _format_graph_result(
        self,
        raw_result: Dict[str, Any],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        格式化图谱检索结果
        
        Args:
            raw_result: 原始图谱结果
            query: 原始查询（用于计算相关性）
            
        Returns:
            格式化后的结果
        """
        related_props = raw_result.get("related_properties", {})
        related_labels = raw_result.get("related_labels", [])
        
        if not related_props:
            return None
        
        # 确定节点类型
        node_type = "Unknown"
        for label in related_labels:
            if label in ["Organization", "Regulation", "AuditCase", "RiskEvent"]:
                node_type = label
                break
        
        # 构建路径描述
        relationships = raw_result.get("relationships", [])
        path_desc = f"{raw_result.get('start_id')} -> {raw_result.get('related_id')}"
        if relationships:
            rel_types = [r.get("type", "") for r in relationships]
            path_desc += f" (via {' -> '.join(rel_types)})"
        
        # 计算简单的相关性分数（基于标签匹配）
        # TODO: 可用更复杂的语义相似度计算
        relevance_score = 0.5  # 默认基础分
        query_lower = query.lower()
        for key, value in related_props.items():
            if isinstance(value, str) and value.lower() in query_lower:
                relevance_score += 0.1
        relevance_score = min(relevance_score, 1.0)
        
        return {
            "type": node_type,
            "node_id": raw_result.get("related_id"),
            "properties": related_props,
            "path_description": path_desc,
            "nodes": [raw_result.get("start_id"), raw_result.get("related_id")],
            "relevance_score": relevance_score,
            "source": "graph",
        }
    
    def _semantic_match_entity(self, entity_text: str, node_text: str) -> float:
        """
        计算实体文本与节点文本的语义相似度
        
        Args:
            entity_text: 实体文本
            node_text: 节点文本
            
        Returns:
            相似度分数 (0.0-1.0)
        """
        if not entity_text or not node_text:
            return 0.0
        
        # 简化的相似度计算
        entity_lower = entity_text.lower()
        node_lower = node_text.lower()
        
        # 完全匹配
        if entity_lower == node_lower:
            return 1.0
        
        # 包含关系
        if entity_lower in node_lower or node_lower in entity_lower:
            return 0.8
        
        # 公共子串长度比例
        import difflib
        matcher = difflib.SequenceMatcher(None, entity_lower, node_lower)
        return matcher.ratio()
    
    def _deduplicate_graph_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        去重图谱结果
        
        Args:
            results: 原始结果列表
            
        Returns:
            去重后的结果列表
        """
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            node_id = result.get("node_id")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                deduplicated.append(result)
        
        # 按相关性分数排序
        deduplicated.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return deduplicated
    
    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        融合向量和图谱结果
        
        采用加权评分融合策略：
        final_score = weight_vector * vector_score + weight_graph * graph_score
        
        Args:
            vector_results: 向量检索结果
            graph_results: 图谱检索结果
            
        Returns:
            字典包含：
            - ``combined_ranking``：列表，每项含 ``source``（``vector``/``graph``）、``content``、
              ``metadata``、``original_score``、``fused_score``（原始分乘权重）、``rank``（分支内序号）、
              ``final_rank``（合并排序后序号）
            - ``vector_weight`` / ``graph_weight``：使用的权重
            - ``total_items``：合并后条目总数
        """
        w_vector = self._fusion_weights.get("vector", 0.6)
        w_graph = self._fusion_weights.get("graph", 0.4)
        
        combined_ranking = []
        
        # 处理向量结果
        for i, result in enumerate(vector_results):
            score = result.get("score", 0.0)
            combined_ranking.append({
                "source": "vector",
                "content": result.get("chunk", ""),
                "metadata": result.get("metadata", {}),
                "original_score": score,
                "fused_score": score * w_vector,
                "rank": i + 1,
            })
        
        # 处理图谱结果
        for i, result in enumerate(graph_results):
            score = result.get("relevance_score", 0.0)
            combined_ranking.append({
                "source": "graph",
                "content": str(result.get("properties", {})),
                "metadata": {
                    "node_id": result.get("node_id"),
                    "type": result.get("type"),
                    "path": result.get("path_description"),
                },
                "original_score": score,
                "fused_score": score * w_graph,
                "rank": i + 1,
            })
        
        # 按融合分数排序
        combined_ranking.sort(key=lambda x: x.get("fused_score", 0), reverse=True)
        
        # 重新分配排名
        for i, item in enumerate(combined_ranking):
            item["final_rank"] = i + 1
        
        return {
            "combined_ranking": combined_ranking,
            "vector_weight": w_vector,
            "graph_weight": w_graph,
            "total_items": len(combined_ranking),
        }


def get_hybrid_retriever(
    vector_top_k: int = None,
    graph_hops: int = None,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> HybridRetriever:
    """
    工厂函数：创建混合检索器实例
    
    Args:
        vector_top_k: 向量检索 TopK
        graph_hops: 图谱跳跃层数
        fusion_weights: 融合权重
        
    Returns:
        HybridRetriever 实例
    """
    return HybridRetriever(
        vector_top_k=vector_top_k,
        graph_hops=graph_hops,
        fusion_weights=fusion_weights,
    )
