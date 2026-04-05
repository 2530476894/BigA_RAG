"""
Configuration Module - 配置管理模块

用途：从环境变量与可选 ``.env`` 文件加载配置，提供全局 ``Settings`` 单例。
关键依赖：pydantic-settings（``BaseSettings``）、python-dotenv（由 pydantic-settings 读取 ``.env``）。
审计场景映射：LLM 参数、Neo4j 连接、向量库配置、RAG 检索策略参数。
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    应用配置类：字段对应环境变量（不区分大小写），可被项目根目录 ``.env`` 覆盖。

    分组概览：``llm_*`` 大模型调用；``neo4j_*`` 图数据库；``chroma_*`` / ``embedding_*`` 向量与嵌入；
    ``vector_top_k`` / ``graph_hops`` / ``fusion_weight_*`` 混合检索与融合；``log_level`` / ``environment`` 运行态；
    ``base_dir`` / ``data_dir`` 等路径与样例数据位置。
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==================== LLM Configuration ====================
    llm_provider: str = Field(default="qwen", description="LLM 提供商")
    llm_model: str = Field(default="qwen-plus", description="LLM 模型名称")
    llm_api_key: str = Field(default="", description="LLM API 密钥 (DASHSCOPE_API_KEY)")
    llm_base_url: Optional[str] = Field(default=None, description="LLM API 基础 URL")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="LLM 温度参数")
    
    # ==================== DashScope Configuration ====================
    dashscope_api_key: str = Field(default="", description="阿里云 DashScope API Key")
    
    # ==================== Neo4j Configuration ====================
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j 连接 URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field(default="audit_password_2024", description="Neo4j 密码")
    neo4j_database: str = Field(default="neo4j", description="Neo4j 数据库名")
    
    # ==================== Vector Database Configuration ====================
    chroma_persist_dir: str = Field(default="./data/chroma_db", description="Chroma 持久化目录")
    chroma_collection: str = Field(default="audit_embeddings", description="Chroma 集合名")
    
    # ==================== Embedding Configuration ====================
    embedding_model: str = Field(default="text-embedding-v3", description="嵌入模型")
    embedding_dimension: int = Field(default=1536, description="嵌入维度")
    
    # ==================== RAG Configuration ====================
    vector_top_k: int = Field(default=5, ge=1, le=20, description="向量检索 TopK")
    graph_hops: int = Field(default=2, ge=1, le=5, description="图谱多跳层数")
    fusion_weight_vector: float = Field(default=0.6, ge=0.0, le=1.0, description="向量结果权重")
    fusion_weight_graph: float = Field(default=0.4, ge=0.0, le=1.0, description="图谱结果权重")
    
    # ==================== Application Settings ====================
    log_level: str = Field(default="INFO", description="日志级别")
    environment: str = Field(default="development", description="运行环境")
    
    # ==================== Paths ====================
    base_dir: str = Field(default=".", description="项目根目录")
    data_dir: str = Field(default="./data", description="数据目录")
    sample_data_dir: str = Field(default="./data/sample", description="样例数据目录")
    mock_triples_path: str = Field(default="./data/mock_triples.json", description="预置三元组路径")
    
    @property
    def neo4j_auth(self) -> tuple:
        """返回 Neo4j 认证元组"""
        return (self.neo4j_username, self.neo4j_password)
    
    @property
    def fusion_weights(self) -> dict:
        """返回融合权重字典，供 ``HybridRetriever._fuse_results`` 对向量与图谱分支加权使用。"""
        return {
            "vector": self.fusion_weight_vector,
            "graph": self.fusion_weight_graph
        }


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """
    获取配置单例
    用于依赖注入
    """
    return settings
