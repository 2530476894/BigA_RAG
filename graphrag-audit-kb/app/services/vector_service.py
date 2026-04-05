"""
Vector Service - 向量数据库服务

用途：提供 Chroma 向量库的索引/查询封装
关键依赖：chromadb
审计场景映射：文档片段向量化存储、相似度检索、混合检索的向量部分
可扩展性：预留 Milvus 等其他向量库的适配接口

注意：模块末尾 ``vector_service`` 在 import 时即构造 ``VectorService``（急加载单例），
与 ``get_neo4j_service`` 的延迟初始化不同。
"""

from typing import Optional, List, Dict, Any
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("vector_service")


class VectorService:
    """
    向量数据库服务类：封装 ChromaDB 持久化客户端与集合；``add_documents`` / ``similarity_search``
    在未接入自定义嵌入模型时依赖 Chroma 默认嵌入与查询管线（见实现内日志）。
    """
    
    _instance: Optional["VectorService"] = None
    
    def __new__(cls) -> "VectorService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.api.models.Collection.Collection] = None
        self._embedding_function: Optional[Any] = None
        
        self._initialize()
        self._initialized = True
        
        logger.info(
            "vector_service_initialized",
            persist_dir=settings.chroma_persist_dir,
            collection=settings.chroma_collection
        )
    
    def _initialize(self):
        """初始化 Chroma 客户端和集合"""
        try:
            # 创建持久化目录
            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            
            # 初始化 Chroma 客户端（持久化模式）
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
            
            # TODO: 初始化嵌入函数（如 OpenAI 兼容 Embeddings API）
            
            logger.info("chroma_client_initialized")
            
        except Exception as e:
            logger.error("chroma_initialization_failed", error=str(e))
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文档到向量库
        
        Args:
            documents: 文档内容列表
            metadatas: 文档元数据列表（可选）
            ids: 文档 ID 列表（可选，若不传则自动生成）
            
        Returns:
            添加的文档 ID 列表

        说明：当前未传入 ``embeddings`` 时由 Chroma 在服务端生成嵌入（与 ``using_placeholder_embeddings`` 警告一致）。
        """
        if not self._collection:
            raise RuntimeError("Vector collection not initialized")
        
        if ids is None:
            # 自动生成 ID
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        try:
            # TODO: 集成真实嵌入模型
            # 当前使用占位方案，需要实现嵌入生成逻辑
            # embeddings = self._embedding_function.embed_documents(documents)
            
            # 临时方案：使用空嵌入（仅用于测试结构）
            # 实际使用时需替换为真实嵌入计算
            logger.warning("using_placeholder_embeddings", count=len(documents))
            
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                # embeddings=embeddings,  # 真实嵌入
            )
            
            logger.info(
                "documents_added",
                count=len(documents),
                collection=settings.chroma_collection
            )
            
            return ids
            
        except Exception as e:
            logger.error("add_documents_failed", error=str(e))
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_metadata: 元数据过滤条件（可选）
            
        Returns:
            检索结果列表，每项包含：chunk, source, score, metadata

        说明：在集合元数据 ``hnsw:space`` 为 cosine 时，将返回的 ``distance`` 近似为 ``1.0 - distance`` 作为 ``score``。
        """
        if not self._collection:
            raise RuntimeError("Vector collection not initialized")
        
        try:
            # TODO: 集成真实嵌入模型
            # query_embedding = self._embedding_function.embed_query(query)
            
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            
            # 格式化结果
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    # 将距离转换为相似度分数（余弦相似度）
                    similarity_score = 1.0 - distance
                    
                    formatted_results.append({
                        "chunk": doc,
                        "source": metadata.get("source", "unknown"),
                        "score": similarity_score,
                        "metadata": metadata,
                    })
            
            logger.info(
                "similarity_search_executed",
                query=query[:50] + "..." if len(query) > 50 else query,
                result_count=len(formatted_results)
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            ids: 要删除的文档 ID 列表
            
        Returns:
            是否删除成功
        """
        if not self._collection:
            raise RuntimeError("Vector collection not initialized")
        
        try:
            self._collection.delete(ids=ids)
            logger.info("documents_deleted", count=len(ids))
            return True
        except Exception as e:
            logger.error("delete_documents_failed", error=str(e))
            return False
    
    def get_document_count(self) -> int:
        """
        获取文档总数
        
        Returns:
            文档数量
        """
        if not self._collection:
            return 0
        
        try:
            return self._collection.count()
        except Exception as e:
            logger.error("get_document_count_failed", error=str(e))
            return 0
    
    def clear_collection(self) -> bool:
        """
        清空集合（慎用）
        
        Returns:
            是否清空成功
        """
        if not self._client:
            return False
        
        try:
            self._client.delete_collection(name=settings.chroma_collection)
            # 重新创建空集合
            self._collection = self._client.create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("collection_cleared")
            return True
        except Exception as e:
            logger.error("clear_collection_failed", error=str(e))
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        try:
            count = self.get_document_count()
            return {
                "status": "healthy",
                "collection": settings.chroma_collection,
                "document_count": count,
            }
        except Exception as e:
            logger.error("vector_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# 模块导入时即创建的单例（供 ``get_vector_service`` 返回）
vector_service = VectorService()


def get_vector_service() -> VectorService:
    """
    获取向量服务单例
    用于依赖注入
    """
    return vector_service
