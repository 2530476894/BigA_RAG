"""
LLM Module - 大语言模型与 Embedding 统一接口抽象

用途：定义 BaseEmbedding 和 BaseLLM 基类，提供 Qwen 实现
关键依赖：dashscope
扩展性：支持后续接入其他 LLM 提供商（OpenAI、Zhipu 等）
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseEmbedding(ABC):
    """Embedding 接口抽象基类"""
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为向量"""
        pass
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量将文档列表转换为向量"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass


class BaseLLM(ABC):
    """LLM 生成接口抽象基类"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """生成文本响应"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """流式生成文本响应"""
        pass


class QwenEmbedding(BaseEmbedding):
    """Qwen Embedding 实现 (text-embedding-v3)"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        self._api_key = api_key
        self._model = model
        self._dimension = 1536  # text-embedding-v3 默认维度
        
        # 延迟导入 dashscope，避免未安装时报错
        try:
            import dashscope
            dashscope.api_key = api_key
            self._dashscope = dashscope
        except ImportError:
            raise ImportError("Please install dashscope: pip install dashscope")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embeddings = await self.embed_documents([text])
        return embeddings[0] if embeddings else []
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        if not texts:
            return []
        
        # 分批处理，避免单次请求过大
        batch_size = 25
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self._call_api(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 DashScope Embedding API"""
        from dashscope import TextEmbedding
        
        try:
            response = TextEmbedding.call(
                model=self._model,
                input=texts,
            )
            
            if response.status_code == 200:
                embeddings = [item["embedding"] for item in response.output["embeddings"]]
                return embeddings
            else:
                raise RuntimeError(
                    f"Qwen Embedding API error: {response.code} - {response.message}"
                )
        except Exception as e:
            raise RuntimeError(f"Qwen Embedding failed: {str(e)}")


class QwenLLM(BaseLLM):
    """Qwen LLM 实现 (qwen-max, qwen-plus, qwen-turbo)"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        
        # 延迟导入 dashscope
        try:
            import dashscope
            dashscope.api_key = api_key
            if base_url:
                dashscope.base_url = base_url
            self._dashscope = dashscope
        except ImportError:
            raise ImportError("Please install dashscope: pip install dashscope")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """同步生成文本响应"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message",
            )
            
            if response.status_code == 200:
                return response.output["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(
                    f"Qwen LLM API error: {response.code} - {response.message}"
                )
        except Exception as e:
            raise RuntimeError(f"Qwen LLM generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """流式生成文本响应"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            from dashscope import Generation
            
            responses = Generation.call(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message",
                stream=True,
                incremental_output=True,
            )
            
            for response in responses:
                if response.status_code == 200:
                    yield response.output["choices"][0]["message"]["content"]
                else:
                    raise RuntimeError(
                        f"Qwen LLM streaming error: {response.code} - {response.message}"
                    )
        except Exception as e:
            raise RuntimeError(f"Qwen LLM streaming failed: {str(e)}")


def create_qwen_embedding(api_key: str, model: str = "text-embedding-v3") -> QwenEmbedding:
    """工厂函数：创建 Qwen Embedding 实例"""
    return QwenEmbedding(api_key=api_key, model=model)


def create_qwen_llm(
    api_key: str,
    model: str = "qwen-plus",
    base_url: Optional[str] = None,
) -> QwenLLM:
    """工厂函数：创建 Qwen LLM 实例"""
    return QwenLLM(api_key=api_key, model=model, base_url=base_url)
