"""
FastAPI Main Entry Point - FastAPI 主入口

用途：定义 API 路由、启动事件、健康检查端点
关键依赖：fastapi, uvicorn
审计场景映射：RAG 查询接口、图谱管理接口、健康监控
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings, get_settings, Settings
from app.utils.logger import setup_logger, get_logger, AuditLogContext
from app.models.schema import RAGQueryRequest, RAGQueryResponse
from app.services.neo4j_service import get_neo4j_service
from app.services.vector_service import get_vector_service
from app.core.retriever import get_hybrid_retriever
from app.core.generator import get_generator, RAGGenerator
from app.llm import create_qwen_llm, QwenLLM

# 初始化日志
setup_logger()
logger = get_logger("main")

# 全局 LLM 客户端和生成器实例
_llm_client: Optional[QwenLLM] = None
_generator: Optional[RAGGenerator] = None


def initialize_llm_components() -> None:
    """在应用启动时初始化 LLM 客户端和生成器"""
    global _llm_client, _generator
    
    # 检查是否配置了 DashScope API Key
    api_key = settings.dashscope_api_key or settings.llm_api_key
    
    if api_key and api_key.strip():
        try:
            _llm_client = create_qwen_llm(
                api_key=api_key.strip(),
                model=settings.llm_model,
                base_url=settings.llm_base_url,
            )
            _generator = get_generator(llm_client=_llm_client)
            logger.info(
                "llm_initialized",
                model=settings.llm_model,
                provider=settings.llm_provider,
            )
        except Exception as e:
            logger.error("llm_initialization_failed", error=str(e))
            _generator = get_generator()  # 降级为无 LLM 模式
    else:
        logger.warning("llm_not_configured", message="DASHSCOPE_API_KEY or LLM_API_KEY not set, running in fallback mode")
        _generator = get_generator()  # 无 LLM 模式


def get_llm_generator() -> RAGGenerator:
    """获取已初始化的生成器实例"""
    if _generator is None:
        initialize_llm_components()
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    应用生命周期管理。

    进入时在启动阶段记录日志并对 Neo4j、Chroma 做预连接健康检查（失败仅告警，不阻塞启动），
    同时初始化 LLM 客户端和生成器。``yield`` 之后应用处于运行期，接受请求；
    进程退出或关闭时在 ``yield`` 之后执行收尾日志。
    """
    # 启动时执行
    logger.info("application_starting", version="0.1.0")
    
    # 初始化 LLM 组件
    initialize_llm_components()
    
    # 初始化服务（预连接检查）
    try:
        neo4j_service = get_neo4j_service()
        health = neo4j_service.health_check()
        logger.info("neo4j_health_check", status=health.get("status"))
    except Exception as e:
        logger.warning("neo4j_startup_check_failed", error=str(e))
    
    try:
        vector_service = get_vector_service()
        health = vector_service.health_check()
        logger.info("chroma_health_check", status=health.get("status"))
    except Exception as e:
        logger.warning("chroma_startup_check_failed", error=str(e))
    
    yield
    
    # 关闭时执行
    logger.info("application_shutting_down")


# 创建 FastAPI 应用
app = FastAPI(
    title="GraphRAG Audit Knowledge Base",
    description="基于知识图谱的审计 RAG 知识库 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Endpoints ====================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    健康检查端点。

    Returns:
        含 ``status`` 与 ``services``。当且仅当 Neo4j 与 Chroma 均报告 healthy 时，
        顶层 ``status`` 为 ``healthy``；任一不可用则为 ``degraded``，并在 ``services`` 中分别给出子状态。
    """
    neo4j_healthy = False
    chroma_healthy = False
    
    try:
        neo4j_service = get_neo4j_service()
        neo4j_health = neo4j_service.health_check()
        neo4j_healthy = neo4j_health.get("status") == "healthy"
    except Exception as e:
        logger.error("neo4j_health_check_failed", error=str(e))
    
    try:
        vector_service = get_vector_service()
        chroma_health = vector_service.health_check()
        chroma_healthy = chroma_health.get("status") == "healthy"
    except Exception as e:
        logger.error("chroma_health_check_failed", error=str(e))
    
    overall_healthy = neo4j_healthy and chroma_healthy
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "services": {
            "neo4j": {
                "status": "healthy" if neo4j_healthy else "unhealthy",
            },
            "chroma": {
                "status": "healthy" if chroma_healthy else "unhealthy",
            },
        },
    }


@app.get("/health/entity", tags=["Health"])
async def entity_health_check():
    """
    实体识别服务健康检查端点。

    Returns:
        实体识别服务的状态信息
    """
    try:
        from app.services.llm_entity_service import get_llm_entity_service
        entity_service = get_llm_entity_service()
        
        # 测试实体提取
        test_entities = await entity_service.extract_entities("测试审计实体识别")
        
        return {
            "status": "healthy",
            "entity_service": {
                "llm_available": entity_service._llm_client is not None,
                "test_extraction": len(test_entities) >= 0,  # 至少不报错
                "extracted_count": len(test_entities)
            }
        }
    except Exception as e:
        logger.error("entity_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "entity_service": {
                "error": str(e)
            }
        }


@app.get("/", tags=["Root"])
async def root():
    """
    根路径欢迎信息
    """
    return {
        "message": "GraphRAG Audit Knowledge Base API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ==================== RAG Query Endpoint ====================

@app.post("/api/v1/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(
    request: RAGQueryRequest,
    settings: Settings = Depends(get_settings),
):
    """
    RAG 查询接口：基于知识图谱和向量检索的审计合规问答。

    处理步骤：1) 生成短 ``task_id``；2) 使用 ``AuditLogContext`` 包裹请求链路以便审计追踪；
    3) 混合检索；4) 基于检索结果生成响应（当前为无 LLM 占位生成，见 ``RAGGenerator``）。

    Args:
        request: RAG 查询请求

    Returns:
        ``RAGQueryResponse``：依据条款、关联案例、置信度、溯源路径等（部分字段随生成器实现可能为空列表）。
    """
    # 生成任务 ID 用于审计追踪
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    with AuditLogContext(task_id=task_id, operation="rag_query"):
        try:
            logger.info(
                "rag_query_received",
                task_id=task_id,
                question=request.question[:50] + "..." if len(request.question) > 50 else request.question,
                vector_top_k=request.vector_top_k,
                graph_hops=request.graph_hops,
            )
            
            # Step 1: 混合检索
            retriever = get_hybrid_retriever(
                vector_top_k=request.vector_top_k,
                graph_hops=request.graph_hops,
                fusion_weights=settings.fusion_weights,
            )
            
            retrieval_results = await retriever.retrieve(
                query=request.question,
                include_cases=request.include_cases,
                include_regulations=request.include_regulations,
            )

            # Step 2: 基于检索生成响应（使用已初始化的 LLM 生成器）
            generator = get_llm_generator()
            response = await generator.generate(
                question=request.question,
                retrieval_results=retrieval_results,
            )
            
            logger.info(
                "rag_query_completed",
                task_id=task_id,
                confidence=response.confidence_score,
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "rag_query_failed",
                task_id=task_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RAG query failed: {str(e)}",
            )


# ==================== Graph Management Endpoints ====================

@app.get("/api/v1/graph/stats", tags=["Graph"])
async def get_graph_stats():
    """
    获取图谱统计信息（节点总数、标签种数、有向关系总数、关系类型种数）。
    """
    try:
        neo4j_service = get_neo4j_service()
        
        # 统计所有节点实例个数及不同标签组合的种类数（distinct labels）
        stats_query = """
        MATCH ()
        RETURN 
            count(*) AS total_nodes,
            count(DISTINCT labels(())) AS label_count
        """
        node_stats = neo4j_service.execute_cypher(stats_query)
        
        # 统计有向关系条数及不同关系类型的种类数
        rel_query = """
        MATCH ()-[r]->()
        RETURN 
            count(r) AS total_relationships,
            count(DISTINCT type(r)) AS relationship_type_count
        """
        rel_stats = neo4j_service.execute_cypher(rel_query)
        
        return {
            "total_nodes": node_stats[0]["total_nodes"] if node_stats else 0,
            "label_count": node_stats[0]["label_count"] if node_stats else 0,
            "total_relationships": rel_stats[0]["total_relationships"] if rel_stats else 0,
            "relationship_type_count": rel_stats[0]["relationship_type_count"] if rel_stats else 0,
        }
    except Exception as e:
        logger.error("get_graph_stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph stats: {str(e)}",
        )


@app.post("/api/v1/graph/init-schema", tags=["Graph"])
async def initialize_graph_schema():
    """
    初始化图谱 Schema（幂等执行 Cypher：创建约束与索引）。

    注意：仅需在首次部署或变更 Schema 后调用；副作用为在 Neo4j 中执行 DDL，已存在时可能跳过并记日志。

    Returns:
        成功消息与 ``status: success`` 的 JSON 对象。
    """
    try:
        neo4j_service = get_neo4j_service()
        neo4j_service.initialize_schema()
        
        return {
            "message": "Graph schema initialized successfully",
            "status": "success",
        }
    except Exception as e:
        logger.error("initialize_schema_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize schema: {str(e)}",
        )


# ==================== Vector Management Endpoints ====================

@app.get("/api/v1/vector/stats", tags=["Vector"])
async def get_vector_stats():
    """
    获取向量库统计信息。

    Returns:
        ``VectorService.health_check()`` 的字典（如 ``status``、``collection``、``document_count`` 等）。
    """
    try:
        vector_service = get_vector_service()
        health = vector_service.health_check()
        
        return health
    except Exception as e:
        logger.error("get_vector_stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector stats: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
