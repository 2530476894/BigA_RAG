"""
LLM Entity Service - LLM驱动的实体识别和链接服务

用途：使用Qwen LLM进行审计领域实体提取和图谱链接
关键依赖：QwenLLM客户端
审计场景映射：从查询中识别组织、法规、案例等实体，并链接到知识图谱
"""

from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import get_logger
from app.utils.prompts import build_entity_extraction_prompt, build_entity_linking_prompt, AUDIT_ENTITY_EXTRACTION_PROMPT, AUDIT_ENTITY_LINKING_PROMPT
from app.llm import create_qwen_llm, QwenLLM

logger = get_logger("llm_entity_service")


class LLMEntityService:
    """
    LLM实体服务类：使用Qwen LLM进行实体识别和链接
    """

    def __init__(self, llm_client: Optional[QwenLLM] = None):
        """
        初始化LLM实体服务

        Args:
            llm_client: QwenLLM客户端实例，如果为None则创建新的
        """
        self._llm_client = llm_client or self._create_llm_client()
        logger.info("llm_entity_service_initialized", has_llm=self._llm_client is not None)

    def _create_llm_client(self) -> Optional[QwenLLM]:
        """创建QwenLLM客户端"""
        api_key = settings.dashscope_api_key or settings.llm_api_key
        if api_key and api_key.strip():
            try:
                return create_qwen_llm(
                    api_key=api_key.strip(),
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                )
            except Exception as e:
                logger.error("llm_client_creation_failed", error=str(e))
                return None
        else:
            logger.warning("llm_api_key_not_configured")
            return None

    async def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        使用LLM从查询中提取审计领域实体

        Args:
            query: 用户查询文本

        Returns:
            实体列表，每个实体包含type, text, confidence
        """
        if not self._llm_client:
            logger.warning("llm_client_not_available_for_entity_extraction")
            return []

        try:
            prompt = build_entity_extraction_prompt(query)
            response = await self._llm_client.generate(
                prompt=prompt,
                system_prompt=AUDIT_ENTITY_EXTRACTION_PROMPT,
                temperature=settings.entity_extraction.get("temperature", 0.1),
                max_tokens=settings.entity_extraction.get("max_tokens", 512),
            )

            entities = self._parse_entity_extraction_response(response)
            logger.info("entity_extraction_completed", query=query[:50], entity_count=len(entities))
            return entities

        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            return []

    async def link_entities(self, entities: List[Dict[str, Any]], graph_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将提取的实体与图谱节点进行语义匹配和链接

        Args:
            entities: 提取的实体列表
            graph_context: 图谱上下文信息

        Returns:
            链接结果列表，每个包含entity, matched_nodes, best_match
        """
        if not self._llm_client:
            logger.warning("llm_client_not_available_for_entity_linking")
            return []

        linked_entities = []

        for entity in entities:
            try:
                prompt = build_entity_linking_prompt(entity, graph_context)
                response = await self._llm_client.generate(
                    prompt=prompt,
                    system_prompt=AUDIT_ENTITY_LINKING_PROMPT,
                    temperature=settings.entity_linking.get("temperature", 0.1),
                    max_tokens=settings.entity_linking.get("max_tokens", 512),
                )

                linked_entity = self._parse_entity_linking_response(entity, response)
                linked_entities.append(linked_entity)

            except Exception as e:
                logger.error("entity_linking_failed", entity=entity, error=str(e))
                linked_entities.append({
                    "entity": entity,
                    "matched_nodes": [],
                    "best_match": None,
                    "linking_confidence": 0.0
                })

        logger.info("entity_linking_completed", entity_count=len(linked_entities))
        return linked_entities

    def _parse_entity_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """
        解析LLM的实体提取响应

        Args:
            response: LLM响应文本

        Returns:
            结构化的实体列表
        """
        entities = []
        try:
            # 简单解析，假设LLM返回JSON格式
            import json
            parsed = json.loads(response)
            if isinstance(parsed, list):
                entities = parsed
            elif isinstance(parsed, dict) and "entities" in parsed:
                entities = parsed["entities"]
        except json.JSONDecodeError:
            # 回退到文本解析
            entities = self._parse_entities_from_text(response)

        # 验证和标准化实体格式
        validated_entities = []
        for entity in entities:
            if isinstance(entity, dict) and "type" in entity and "text" in entity:
                validated_entity = {
                    "type": entity["type"],
                    "text": entity["text"],
                    "confidence": entity.get("confidence", 0.8)
                }
                validated_entities.append(validated_entity)

        return validated_entities

    def _parse_entities_from_text(self, response: str) -> List[Dict[str, Any]]:
        """
        从文本响应中解析实体（备用方法）

        Args:
            response: 文本响应

        Returns:
            实体列表
        """
        entities = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_type = parts[0].strip().lower()
                    entity_text = parts[1].strip()
                    if entity_type in ["organization", "regulation", "case", "risk_event"]:
                        entities.append({
                            "type": entity_type,
                            "text": entity_text,
                            "confidence": 0.7
                        })
        return entities

    def _parse_entity_linking_response(self, entity: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        解析实体链接响应

        Args:
            entity: 原始实体
            response: LLM响应

        Returns:
            链接结果
        """
        try:
            import json
            parsed = json.loads(response)
            return {
                "entity": entity,
                "matched_nodes": parsed.get("matched_nodes", []),
                "best_match": parsed.get("best_match"),
                "linking_confidence": parsed.get("confidence", 0.0)
            }
        except json.JSONDecodeError:
            return {
                "entity": entity,
                "matched_nodes": [],
                "best_match": None,
                "linking_confidence": 0.0
            }


def get_llm_entity_service() -> LLMEntityService:
    """
    获取LLM实体服务实例（单例）

    Returns:
        LLMEntityService实例
    """
    if not hasattr(get_llm_entity_service, "_instance"):
        get_llm_entity_service._instance = LLMEntityService()
    return get_llm_entity_service._instance