"""
Neo4j Service - 图数据库服务

用途：提供 Neo4j 连接池管理和基础 CRUD/Cypher 执行能力
关键依赖：neo4j driver
审计场景映射：图谱节点/关系写入、Cypher 查询、多跳关联检索
健壮性：连接池管理、异常处理、事务支持
"""

from typing import Optional, List, Dict, Any, Generator
from contextlib import contextmanager
import neo4j
from neo4j import GraphDatabase, Session, Transaction
from app.config import settings
from app.utils.logger import get_logger
from app.models.kg_schema import generate_cypher_constraints, generate_cypher_indexes

logger = get_logger("neo4j_service")


class Neo4jService:
    """
    Neo4j 图数据库服务类
    采用单例模式，提供连接池管理和 Cypher 执行接口
    """
    
    _instance: Optional["Neo4jService"] = None
    
    def __new__(cls) -> "Neo4jService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._driver: Optional[neo4j.Driver] = None
        self._database: str = settings.neo4j_database
        self._connect()
        self._initialized = True
        
        logger.info(
            "neo4j_service_initialized",
            uri=settings.neo4j_uri,
            database=self._database
        )
    
    def _connect(self):
        """建立 Neo4j 连接"""
        try:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=settings.neo4j_auth,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30.0,
            )
            # 验证连接
            self._driver.verify_connectivity()
            logger.info("neo4j_connection_established")
        except Exception as e:
            logger.error("neo4j_connection_failed", error=str(e))
            raise
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            logger.info("neo4j_connection_closed")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        获取数据库会话上下文管理器
        
        Yields:
            Neo4j Session 对象
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        session = self._driver.session(database=self._database)
        try:
            yield session
        finally:
            session.close()
    
    @contextmanager
    def transaction(self, session: Session) -> Generator[Transaction, None, None]:
        """
        获取事务上下文管理器
        
        Args:
            session: Neo4j Session 对象
            
        Yields:
            Neo4j Transaction 对象
        """
        tx = session.begin_transaction()
        try:
            yield tx
            tx.commit()
        except Exception as e:
            tx.rollback()
            logger.error("neo4j_transaction_rolled_back", error=str(e))
            raise
    
    # ==================== Node Operations ====================
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """
        创建单个节点
        
        Args:
            label: 节点 Label
            properties: 节点属性字典
            
        Returns:
            节点 ID
        """
        query = f"""
        CREATE (n:{label} $properties)
        RETURN n.id AS id
        """
        
        with self.session() as session:
            result = session.run(query, properties=properties)
            record = result.single()
            node_id = record["id"] if record else properties.get("id")
            
            logger.info(
                "node_created",
                label=label,
                node_id=node_id
            )
            return node_id
    
    def create_nodes_batch(self, label: str, properties_list: List[Dict[str, Any]]) -> int:
        """
        批量创建节点
        
        Args:
            label: 节点 Label
            properties_list: 节点属性列表
            
        Returns:
            创建的节点数量
        """
        query = f"""
        UNWIND $properties_list AS props
        CREATE (n:{label} $props)
        RETURN count(n) AS created_count
        """
        
        with self.session() as session:
            result = session.run(query, properties_list=properties_list)
            record = result.single()
            count = record["created_count"] if record else 0
            
            logger.info(
                "nodes_batch_created",
                label=label,
                count=count
            )
            return count
    
    def get_node_by_id(self, label: str, node_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 查询节点
        
        Args:
            label: 节点 Label
            node_id: 节点 ID
            
        Returns:
            节点属性字典，不存在则返回 None
        """
        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        RETURN properties(n) AS properties
        """
        
        with self.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record:
                return dict(record["properties"])
            return None
    
    def delete_node(self, label: str, node_id: str) -> bool:
        """
        删除节点（包括相关关系）
        
        Args:
            label: 节点 Label
            node_id: 节点 ID
            
        Returns:
            是否删除成功
        """
        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """
        
        with self.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            deleted = record["deleted_count"] > 0 if record else False
            
            if deleted:
                logger.info("node_deleted", label=label, node_id=node_id)
            return deleted
    
    # ==================== Relationship Operations ====================
    
    def create_relationship(
        self,
        source_label: str,
        source_id: str,
        target_label: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        创建关系
        
        Args:
            source_label: 源节点 Label
            source_id: 源节点 ID
            target_label: 目标节点 Label
            target_id: 目标节点 ID
            rel_type: 关系类型
            properties: 关系属性（可选）
            
        Returns:
            是否创建成功
        """
        query = f"""
        MATCH (source:{source_label} {{id: $source_id}})
        MATCH (target:{target_label} {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        {'SET r += $properties' if properties else ''}
        RETURN count(r) AS created_count
        """
        
        with self.session() as session:
            result = session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=properties or {}
            )
            record = result.single()
            created = record["created_count"] > 0 if record else False
            
            if created:
                logger.info(
                    "relationship_created",
                    source=f"{source_label}:{source_id}",
                    target=f"{target_label}:{target_id}",
                    type=rel_type
                )
            return created
    
    # ==================== Query Operations ====================
    
    def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            records = []
            for record in result:
                records.append(dict(record))
            return records
    
    def multi_hop_query(
        self,
        start_label: str,
        start_id: str,
        hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        多跳关联查询
        
        Args:
            start_label: 起始节点 Label
            start_id: 起始节点 ID
            hops: 跳跃层数
            relationship_types: 限制的关系类型列表（可选）
            
        Returns:
            关联子图数据
        """
        # 构建可变长度的关系路径
        rel_pattern = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_pattern = f"-[:{rel_types}*1..{hops}]"
        else:
            rel_pattern = f"-[*1..{hops}]"
        
        query = f"""
        MATCH (start:{start_label} {{id: $start_id}})
        MATCH path = (start){rel_pattern}(related)
        RETURN 
            start.id AS start_id,
            related.id AS related_id,
            labels(related) AS related_labels,
            properties(related) AS related_properties,
            relationships(path) AS rels
        LIMIT 50
        """
        
        results = self.execute_cypher(query, {"start_id": start_id})
        
        # 格式化结果
        formatted_results = []
        for record in results:
            formatted_results.append({
                "start_id": record["start_id"],
                "related_id": record["related_id"],
                "related_labels": list(record["related_labels"]),
                "related_properties": dict(record["related_properties"]),
                "relationships": [
                    {
                        "type": rel.type,
                        "direction": "outgoing"
                    }
                    for rel in record["rels"]
                ]
            })
        
        logger.info(
            "multi_hop_query_executed",
            start=f"{start_label}:{start_id}",
            hops=hops,
            result_count=len(formatted_results)
        )
        
        return formatted_results
    
    def search_nodes(
        self,
        label: str,
        search_field: str,
        search_value: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        模糊搜索节点
        
        Args:
            label: 节点 Label
            search_field: 搜索字段
            search_value: 搜索值
            limit: 返回数量限制
            
        Returns:
            匹配的节点列表
        """
        query = f"""
        MATCH (n:{label})
        WHERE toLower(n.{search_field}) CONTAINS toLower($search_value)
        RETURN properties(n) AS properties
        LIMIT $limit
        """
        
        results = self.execute_cypher(query, {"search_value": search_value, "limit": limit})
        return [dict(r["properties"]) for r in results]
    
    # ==================== Schema Initialization ====================
    
    def initialize_schema(self):
        """
        初始化图谱 schema（创建约束和索引）
        应在首次部署时调用
        """
        logger.info("neo4j_schema_initialization_started")
        
        with self.session() as session:
            # 创建约束
            for stmt in generate_cypher_constraints():
                try:
                    session.run(stmt)
                    logger.info("constraint_created", statement=stmt)
                except Exception as e:
                    logger.warning("constraint_creation_skipped", statement=stmt, error=str(e))
            
            # 创建索引
            for stmt in generate_cypher_indexes():
                try:
                    session.run(stmt)
                    logger.info("index_created", statement=stmt)
                except Exception as e:
                    logger.warning("index_creation_skipped", statement=stmt, error=str(e))
        
        logger.info("neo4j_schema_initialization_completed")
    
    # ==================== Health Check ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        try:
            query = "RETURN 1 AS status"
            result = self.execute_cypher(query)
            is_healthy = len(result) > 0 and result[0].get("status") == 1
            
            # 获取统计信息
            stats_query = """
            MATCH ()
            RETURN 
                count(*) AS total_nodes,
                count(DISTINCT labels(())) AS label_count
            """
            stats = self.execute_cypher(stats_query)
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "total_nodes": stats[0]["total_nodes"] if stats else 0,
                "label_count": stats[0]["label_count"] if stats else 0
            }
        except Exception as e:
            logger.error("neo4j_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 全局服务实例（延迟初始化，避免导入时连接失败）
_neo4j_service_instance = None


def get_neo4j_service() -> Neo4jService:
    """
    获取 Neo4j 服务单例（延迟初始化）
    用于依赖注入
    """
    global _neo4j_service_instance
    if _neo4j_service_instance is None:
        try:
            _neo4j_service_instance = Neo4jService()
        except Exception as e:
            logger.warning("neo4j_service_init_failed", error=str(e), note="Will retry on next call")
            raise
    return _neo4j_service_instance


# 兼容性导出（已废弃，建议使用 get_neo4j_service()）
neo4j_service = None  # type: ignore
    """
    return neo4j_service
