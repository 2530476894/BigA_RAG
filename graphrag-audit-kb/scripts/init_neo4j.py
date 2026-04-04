#!/usr/bin/env python3
"""
Neo4j Data Initialization Script

用途：将 mock_triples.json 批量写入 Neo4j 图数据库
关键依赖：neo4j driver, python-dotenv
审计场景映射：初始化审计知识图谱的测试数据，包括组织、法规、案例、风险事件等

使用方法:
    python scripts/init_neo4j.py
    
环境变量:
    NEO4J_URI: Neo4j 连接地址 (默认：bolt://localhost:7687)
    NEO4J_USER: Neo4j 用户名 (默认：neo4j)
    NEO4J_PASSWORD: Neo4j 密码 (默认：audit_password_2024)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(project_root / ".env")

# 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "audit_password_2024")
DATA_FILE = project_root / "data" / "mock_triples.json"


class Neo4jInitializer:
    """Neo4j 数据初始化器"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化 Neo4j 驱动器
        
        Args:
            uri: Neo4j 连接 URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()
    
    def _verify_connection(self):
        """验证 Neo4j 连接"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✓ 成功连接到 Neo4j: {NEO4J_URI}")
        except Exception as e:
            print(f"✗ 连接 Neo4j 失败：{e}")
            raise
    
    def close(self):
        """关闭驱动器"""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j 连接已关闭")
    
    def clear_existing_data(self):
        """清空现有数据（可选）"""
        print("\n⚠  正在清空现有数据...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ 现有数据已清空")
    
    def create_constraints(self):
        """创建唯一性约束"""
        print("\n📋 正在创建唯一性约束...")
        
        constraints = [
            # Organization 节点约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            # Regulation 节点约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.id IS UNIQUE",
            # AuditCase 节点约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:AuditCase) REQUIRE c.id IS UNIQUE",
            # RiskEvent 节点约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:RiskEvent) REQUIRE e.id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"  ✓ 创建约束：{constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    print(f"  ! 约束已存在或创建失败：{e}")
        
        print("✓ 唯一性约束创建完成")
    
    def load_nodes(self, nodes: List[Dict[str, Any]]):
        """
        批量加载节点
        
        Args:
            nodes: 节点列表，每个节点包含 label 和 properties
        """
        print(f"\n📦 正在加载 {len(nodes)} 个节点...")
        
        with self.driver.session() as session:
            for node in nodes:
                label = node.get("label", "")
                props = node.get("properties", {})
                
                if not label or not props:
                    continue
                
                # 构建 Cypher 语句
                # 使用 MERGE 避免重复插入
                prop_keys = list(props.keys())
                prop_values = list(props.values())
                
                # 查找用于 MERGE 的唯一标识字段
                merge_field = "id" if "id" in props else prop_keys[0]
                merge_value = props[merge_field]
                
                # 构建 SET 子句
                set_clauses = ", ".join([f"n.{k} = ${k}" for k in prop_keys])
                
                cypher = f"""
                MERGE (n:{label} {{{merge_field}: ${merge_field}}})
                SET {set_clauses}
                """
                
                try:
                    session.run(cypher, **props)
                    print(f"  ✓ 节点：{label} - {props.get('name', props.get('title', merge_value))}")
                except Exception as e:
                    print(f"  ✗ 节点插入失败：{label} - {e}")
        
        print(f"✓ 节点加载完成")
    
    def load_relations(self, relations: List[Dict[str, Any]]):
        """
        批量加载关系
        
        Args:
            relations: 关系列表，每个关系包含 source, target, type, properties
        """
        print(f"\n🔗 正在加载 {len(relations)} 个关系...")
        
        with self.driver.session() as session:
            for rel in relations:
                source_id = rel.get("source", "")
                target_id = rel.get("target", "")
                rel_type = rel.get("type", "")
                props = rel.get("properties", {})
                
                if not source_id or not target_id or not rel_type:
                    continue
                
                # 构建 Cypher 语句
                # 先查找源节点和目标节点，然后创建/更新关系
                prop_params = {"source_id": source_id, "target_id": target_id}
                set_clause = ""
                if props:
                    set_items = [f"r.{k} = ${k}" for k in props.keys()]
                    set_clause = "SET " + ", ".join(set_items)
                    prop_params.update(props)
                
                cypher = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                {set_clause}
                """
                
                try:
                    session.run(cypher, **prop_params)
                    print(f"  ✓ 关系：{source_id} -[{rel_type}]-> {target_id}")
                except Exception as e:
                    print(f"  ✗ 关系创建失败：{source_id} -> {target_id} - {e}")
        
        print(f"✓ 关系加载完成")
    
    def initialize_from_file(self, data_file: Path, clear_first: bool = False):
        """
        从 JSON 文件初始化数据
        
        Args:
            data_file: 数据文件路径
            clear_first: 是否先清空现有数据
        """
        print(f"\n🚀 开始从 {data_file} 初始化 Neo4j 数据...")
        
        # 读取 JSON 文件
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        relations = data.get("relations", [])
        
        print(f"  读取到 {len(nodes)} 个节点，{len(relations)} 个关系")
        
        # 可选：清空现有数据
        if clear_first:
            self.clear_existing_data()
        
        # 创建约束
        self.create_constraints()
        
        # 加载节点
        self.load_nodes(nodes)
        
        # 加载关系
        self.load_relations(relations)
        
        # 验证数据
        self.verify_data()
        
        print("\n✅ Neo4j 数据初始化完成！")
    
    def verify_data(self):
        """验证数据"""
        print("\n🔍 正在验证数据...")
        
        with self.driver.session() as session:
            # 统计各类节点数量
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY label
            """)
            
            print("\n  节点统计:")
            for record in result:
                print(f"    {record['label']}: {record['count']}")
            
            # 统计关系数量
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY type
            """)
            
            print("\n  关系统计:")
            for record in result:
                print(f"    {record['type']}: {record['count']}")


def main():
    """主函数"""
    print("=" * 60)
    print("Neo4j 数据初始化工具")
    print("=" * 60)
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"数据文件：{DATA_FILE}")
    print()
    
    # 检查数据文件是否存在
    if not DATA_FILE.exists():
        print(f"✗ 数据文件不存在：{DATA_FILE}")
        sys.exit(1)
    
    # 创建初始化器并执行初始化
    initializer = Neo4jInitializer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        initializer.initialize_from_file(DATA_FILE, clear_first=False)
    except Exception as e:
        print(f"\n✗ 初始化失败：{e}")
        sys.exit(1)
    finally:
        initializer.close()


if __name__ == "__main__":
    main()
