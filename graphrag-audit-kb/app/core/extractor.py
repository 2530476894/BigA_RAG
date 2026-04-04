"""
Extractor Module - LLM 驱动的实体/关系抽取 Pipeline

用途：从审计文档中抽取实体和关系，构建知识图谱
关键依赖：langchain, LLM
审计场景映射：法规条款、审计案例、风险事件、组织机构的自动抽取
可扩展性：预留 LangChain DocumentLoader 接入点，支持多种文档格式
"""

from typing import List, Dict, Any, Optional
import json
from app.utils.logger import get_logger
from app.utils.prompts import entity_extraction_prompt
from app.models.kg_schema import NODE_LABELS, RELATIONSHIP_TYPES

logger = get_logger("extractor")


class EntityRelationExtractor:
    """
    实体关系抽取器
    使用 LLM 从非结构化文本中抽取审计领域实体和关系
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        初始化抽取器
        
        Args:
            llm_client: LLM 客户端实例（LangChain LLM 或兼容接口）
                       若为 None，需后续注入
        """
        self._llm_client = llm_client
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
        从文本中抽取实体和关系
        
        Args:
            text: 输入文本
            entity_types: 限定抽取的实体类型列表（可选）
            relation_types: 限定抽取的关系类型列表（可选）
            
        Returns:
            包含 entities 和 relations 的字典
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized. Call set_llm_client() first.")
        
        # 构建抽取提示
        # TODO: 增强 Prompt，支持 entity_types 和 relation_types 的动态约束
        prompt_value = entity_extraction_prompt.format(text=text)
        
        try:
            # 调用 LLM
            # 假设 llm_client 有 invoke 或 __call__ 方法
            if hasattr(self._llm_client, "invoke"):
                response = await self._llm_client.invoke(prompt_value)
            else:
                response = await self._llm_client(prompt_value)
            
            # 解析响应（期望 JSON 格式）
            result = self._parse_extraction_response(response)
            
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


# 导入 asyncio（在文件顶部可能未定义）
import asyncio


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
