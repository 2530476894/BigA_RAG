"""
Extractor Module - LLM 驱动的实体/关系抽取 Pipeline

用途：从审计文档中抽取实体和关系，构建知识图谱
关键依赖：langchain, LLM
审计场景映射：法规条款、审计案例、风险事件、组织机构的自动抽取
可扩展性：预留 LangChain DocumentLoader 接入点，支持多种文档格式
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
from app.utils.logger import get_logger
from app.utils.prompts import entity_extraction_prompt
from app.models.kg_schema import NODE_LABELS, RELATIONSHIP_TYPES

logger = get_logger("extractor")


# ==================== JSON Schema Output Models (Pydantic v2) ====================
class ExtractedEntity(PydanticBaseModel):
    """抽取的实体模型（用于 LangChain JSON 输出约束）"""
    type: str = PydanticField(description="实体类型：Organization/Regulation/AuditCase/RiskEvent/Person")
    name: str = PydanticField(description="实体名称")
    attributes: Optional[Dict[str, Any]] = PydanticField(default=None, description="实体属性")


class ExtractedRelation(PydanticBaseModel):
    """抽取的关系模型（用于 LangChain JSON 输出约束）"""
    source: str = PydanticField(description="源实体名称")
    target: str = PydanticField(description="目标实体名称")
    type: str = PydanticField(description="关系类型：ISSUED_BY/APPLIES_TO/INVOLVED_IN/DETECTED_IN/VIOLATED/PENALIZED_BY")
    properties: Optional[Dict[str, Any]] = PydanticField(default=None, description="关系属性")


class ExtractionOutput(PydanticBaseModel):
    """抽取输出总模型（用于 LangChain JSON 输出约束）"""
    entities: List[ExtractedEntity] = PydanticField(description="实体列表")
    relations: List[ExtractedRelation] = PydanticField(description="关系列表")


# ==================== Entity Disambiguation Rules ====================
# 实体消歧规则配置
ENTITY_DISAMBIGUATION_RULES = {
    # 同义词映射
    "synonyms": {
        "Organization": {
            "aliases": ["公司", "企业", "单位", "机构", "局", "厅", "部"],
            "canonical_field": "name"
        },
        "Regulation": {
            "aliases": ["法规", "法律", "规章", "规定", "办法", "条例"],
            "canonical_field": "title"
        },
        "AuditCase": {
            "aliases": ["案例", "案件", "审计项目"],
            "canonical_field": "title"
        }
    },
    # 归一化规则
    "normalization": {
        "Organization": lambda x: x.replace("有限责任公司", "有限公司").strip(),
        "Regulation": lambda x: x.replace("《", "").replace("》", "").strip()
    },
    # 消歧优先级：当多个实体同名时，按此优先级选择
    "priority_fields": {
        "Organization": ["registration_code", "id"],
        "Regulation": ["clause_id", "effective_date"],
        "AuditCase": ["case_number", "case_date"]
    }
}


class EntityRelationExtractor:
    """
    实体关系抽取器
    使用 LLM 从非结构化文本中抽取审计领域实体和关系
    支持 JSON Schema 输出约束和实体消歧
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        初始化抽取器
        
        Args:
            llm_client: LLM 客户端实例（LangChain LLM 或兼容接口）
                       若为 None，需后续注入
        """
        self._llm_client = llm_client
        self._parser = JsonOutputParser(pydantic_object=ExtractionOutput)
        logger.info("entity_relation_extractor_initialized")
    
    def set_llm_client(self, llm_client: Any):
        """
        注入 LLM 客户端
        
        Args:
            llm_client: LLM 客户端实例
        """
        self._llm_client = llm_client
        logger.info("llm_client_injected")
    
    async def extract_from_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        从文本中抽取实体和关系（基于 LangChain LLM 三元组抽取链）
        
        Args:
            text: 输入文本
            entity_types: 限定抽取的实体类型列表（可选）
            relation_types: 限定抽取的关系类型列表（可选）
            
        Returns:
            包含 entities 和 relations 的字典
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized. Call set_llm_client() first.")
        
        try:
            # 构建带 JSON Schema 约束的抽取链
            # 使用 with_structured_output 方法强制 JSON 输出
            if hasattr(self._llm_client, "with_structured_output"):
                structured_llm = self._llm_client.with_structured_output(ExtractionOutput)
            else:
                structured_llm = self._llm_client
            
            # 构建提示词
            prompt_value = entity_extraction_prompt.format(text=text)
            
            # 调用 LLM
            if hasattr(structured_llm, "invoke"):
                response = await structured_llm.invoke(prompt_value)
            elif hasattr(structured_llm, "__call__"):
                response = await structured_llm(prompt_value)
            else:
                # 回退到手动解析
                return await self._extract_with_manual_parse(text)
            
            # 将响应转换为标准格式
            result = self._convert_extraction_response(response)
            
            # 应用实体消歧规则
            result = self._apply_disambiguation(result)
            
            # 验证抽取结果
            validated_result = self._validate_extraction(result, entity_types, relation_types)
            
            logger.info(
                "extraction_completed",
                entity_count=len(validated_result.get("entities", [])),
                relation_count=len(validated_result.get("relations", []))
            )
            
            return validated_result
            
        except Exception as e:
            logger.error("extraction_failed", error=str(e))
            return {"entities": [], "relations": [], "error": str(e)}
    
    async def _extract_with_manual_parse(self, text: str) -> Dict[str, Any]:
        """
        回退方案：手动解析 LLM 响应
        """
        prompt_value = entity_extraction_prompt.format(text=text)
        
        if hasattr(self._llm_client, "invoke"):
            response = await self._llm_client.invoke(prompt_value)
        else:
            response = await self._llm_client(prompt_value)
        
        result = self._parse_extraction_response(response)
        result = self._apply_disambiguation(result)
        return result
    
    def _parse_extraction_response(self, response: Any) -> Dict[str, Any]:
        """
        解析 LLM 响应
        
        Args:
            response: LLM 原始响应
            
        Returns:
            解析后的字典
        """
        # 提取响应内容
        content = response
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict) and "text" in response:
            content = response["text"]
        
        # 尝试解析 JSON
        # 处理可能的 markdown 代码块包裹
        content_str = str(content).strip()
        if content_str.startswith("```json"):
            content_str = content_str[7:]
        if content_str.endswith("```"):
            content_str = content_str[:-3]
        content_str = content_str.strip()
        
        try:
            result = json.loads(content_str)
            return result
        except json.JSONDecodeError as e:
            logger.warning("json_parse_failed", error=str(e), content_preview=content_str[:200])
            # 返回空结果而非抛出异常
            return {"entities": [], "relations": []}
    
    def _validate_extraction(
        self,
        result: Dict[str, Any],
        entity_types: Optional[List[str]],
        relation_types: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        验证抽取结果
        
        Args:
            result: 解析后的结果
            entity_types: 允许的实体类型
            relation_types: 允许的关系类型
            
        Returns:
            验证后的结果
        """
        validated_entities = []
        validated_relations = []
        
        # 过滤实体
        allowed_entity_types = set(entity_types) if entity_types else NODE_LABELS
        for entity in result.get("entities", []):
            entity_type = entity.get("type", "")
            if entity_type in allowed_entity_types:
                validated_entities.append(entity)
        
        # 过滤关系
        allowed_relation_types = set(relation_types) if relation_types else RELATIONSHIP_TYPES
        for relation in result.get("relations", []):
            rel_type = relation.get("type", "")
            if rel_type in allowed_relation_types:
                validated_relations.append(relation)
        
        return {
            "entities": validated_entities,
            "relations": validated_relations,
        }
    
    def _convert_extraction_response(self, response: Any) -> Dict[str, Any]:
        """
        将 LangChain 结构化响应转换为标准格式
        
        Args:
            response: LangChain 结构化输出（ExtractionOutput 实例或字典）
            
        Returns:
            标准格式的字典
        """
        if hasattr(response, "dict"):
            # Pydantic 模型
            data = response.dict()
        elif isinstance(response, dict):
            data = response
        else:
            data = {"entities": [], "relations": []}
        
        # 转换为内部格式
        entities = []
        for e in data.get("entities", []):
            entities.append({
                "type": e.get("type", ""),
                "name": e.get("name", ""),
                "attributes": e.get("attributes")
            })
        
        relations = []
        for r in data.get("relations", []):
            relations.append({
                "source": r.get("source", ""),
                "target": r.get("target", ""),
                "type": r.get("type", ""),
                "properties": r.get("properties")
            })
        
        return {"entities": entities, "relations": relations}
    
    def _apply_disambiguation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用实体消歧规则
        
        消歧策略：
        1. 名称归一化：去除冗余后缀、统一格式
        2. 同义词合并：将别名映射到标准名称
        3. 优先级去重：当多个实体同名时，保留优先级字段最完整的
        
        Args:
            result: 抽取结果
            
        Returns:
            消歧后的结果
        """
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        
        # 1. 名称归一化
        normalized_entities = []
        for entity in entities:
            entity_type = entity.get("type", "")
            name = entity.get("name", "")
            
            # 应用归一化规则
            if entity_type in ENTITY_DISAMBIGUATION_RULES["normalization"]:
                normalize_func = ENTITY_DISAMBIGUATION_RULES["normalization"][entity_type]
                name = normalize_func(name)
            
            entity["name"] = name
            normalized_entities.append(entity)
        
        # 2. 去重（基于名称和类型）
        seen = {}
        deduped_entities = []
        for entity in normalized_entities:
            key = f"{entity.get('type', '')}:{entity.get('name', '')}"
            if key not in seen:
                seen[key] = entity
                deduped_entities.append(entity)
            else:
                # 合并属性：保留更丰富的属性
                existing = seen[key]
                existing_attrs = existing.get("attributes") or {}
                new_attrs = entity.get("attributes") or {}
                if len(new_attrs) > len(existing_attrs):
                    seen[key]["attributes"] = new_attrs
        
        # 3. 更新关系中的实体引用
        entity_name_map = {e.get("name"): e.get("name") for e in deduped_entities}
        deduped_relations = []
        for rel in relations:
            source = rel.get("source", "")
            target = rel.get("target", "")
            
            # 检查源和目标是否在去重后的实体列表中
            if any(e.get("name") == source for e in deduped_entities) and \
               any(e.get("name") == target for e in deduped_entities):
                deduped_relations.append(rel)
        
        logger.info(
            "disambiguation_applied",
            before_count=len(entities),
            after_count=len(deduped_entities)
        )
        
        return {"entities": deduped_entities, "relations": deduped_relations}
    
    async def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        批量抽取
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            抽取结果列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.extract_from_text(text) for text in batch],
                return_exceptions=True
            )
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"entities": [], "relations": [], "error": str(result)})
                else:
                    results.append(result)
        
        logger.info("batch_extraction_completed", total=len(results))
        return results


# ==================== Document Loader Integration ====================
# 预留 LangChain DocumentLoader 接入点

class DocumentProcessor:
    """
    文档处理器
    封装文档加载、分块、抽取的完整流程
    """
    
    def __init__(
        self,
        extractor: EntityRelationExtractor,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化文档处理器
        
        Args:
            extractor: 实体关系抽取器
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
        """
        self._extractor = extractor
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        logger.info("document_processor_initialized")
    
    async def process_document(
        self,
        document: Any,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        处理单个文档
        
        Args:
            document: 文档对象（LangChain Document 或文件路径）
            source_metadata: 源元数据
            
        Returns:
            处理结果，包含抽取的实体和关系
        """
        # TODO: 集成 LangChain DocumentLoader
        # 支持以下文档格式：
        # - TextLoader: .txt 文件
        # - DirectoryLoader: 目录批量加载
        # - PyPDFLoader: PDF 文件
        # - Docx2txtLoader: Word 文档
        # - Unstructured 系列加载器
        
        logger.warning("document_loader_not_implemented", todo="Integrate LangChain DocumentLoader")
        
        # 占位实现
        return {
            "entities": [],
            "relations": [],
            "chunks_processed": 0,
        }
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self._chunk_overlap
        return chunks


def get_extractor(llm_client: Optional[Any] = None) -> EntityRelationExtractor:
    """
    工厂函数：创建抽取器实例
    
    Args:
        llm_client: LLM 客户端（可选）
        
    Returns:
        EntityRelationExtractor 实例
    """
    return EntityRelationExtractor(llm_client)


def get_document_processor(
    extractor: EntityRelationExtractor,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> DocumentProcessor:
    """
    工厂函数：创建文档处理器实例
    
    Args:
        extractor: 实体关系抽取器
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        
    Returns:
        DocumentProcessor 实例
    """
    return DocumentProcessor(extractor, chunk_size, chunk_overlap)
