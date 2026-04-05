"""
Logger Utility - 统一日志配置

用途：提供结构化日志记录，支持分级输出和审计追踪
关键依赖：structlog, logging
审计场景映射：操作审计、错误追踪、检索过程记录
"""

import logging
import sys
from typing import Optional
import structlog
from app.config import settings


def setup_logger(log_level: Optional[str] = None) -> logging.Logger:
    """
    配置并返回根日志记录器
    
    Args:
        log_level: 日志级别，默认从配置读取
        
    Returns:
        配置好的 logger 实例
    """
    level = getattr(logging, (log_level or settings.log_level).upper(), logging.INFO)
    
    # 配置标准 logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # 配置 structlog 用于结构化日志
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return logging.getLogger("graphrag_audit")


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    获取绑定上下文的 logger
    
    Args:
        name: 模块名称
        
    Returns:
        structlog 绑定的 logger
    """
    return structlog.get_logger(name)


# 审计专用日志上下文管理器
class AuditLogContext:
    """
    审计日志上下文管理器：进入时绑定 ``task_id``/``operation`` 并打 ``audit_task_started``；
    退出时若无异常打 ``audit_task_completed``，否则打 ``audit_task_failed``。
    """
    
    def __init__(self, task_id: str, operation: str):
        self.task_id = task_id
        self.operation = operation
        self.logger = get_logger("audit_context")
    
    def __enter__(self):
        self.logger = self.logger.bind(
            task_id=self.task_id,
            operation=self.operation
        )
        self.logger.info("audit_task_started", status="started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                "audit_task_failed",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                status="failed"
            )
        else:
            self.logger.info("audit_task_completed", status="completed")
        return False
    
    def log_retrieval(self, query: str, result_count: int, source: str):
        """记录检索操作：``source`` 标识向量/图谱等来源。"""
        self.logger.info(
            "retrieval_executed",
            query=query,
            result_count=result_count,
            source=source
        )
    
    def log_generation(self, prompt_tokens: int, response_tokens: int):
        """记录生成操作：占位 token 计数便于后续接入 LLM 时对齐。"""
        self.logger.info(
            "generation_executed",
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens
        )
    
    def log_validation(self, validation_type: str, passed: bool, details: dict):
        """记录校验操作：``validation_type`` 区分金额/时间等；``details`` 为结构化补充信息。"""
        self.logger.info(
            "validation_executed",
            validation_type=validation_type,
            passed=passed,
            details=details
        )


# 模块级默认 logger（无模块名绑定，供本包内简单引用）
logger = get_logger()
