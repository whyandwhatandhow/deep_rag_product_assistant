# backend/app/retriever/query_rewriter.py
import os
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from langchain_deepseek import ChatDeepSeek
from app.core.config import settings

load_dotenv(override=True)


class QueryRewriter:
    def __init__(
        self,
        strategy: str = "multi-query",
        num_queries: int = 3,
        usehyde: bool = False
    ):
        self.strategy = strategy
        self.num_queries = num_queries
        self.usehyde = usehyde

        api_key = settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = settings.deepseek_base_url or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

        self.llm = None
        if api_key and api_key != "your_api_key_here":
            try:
                self.llm = ChatDeepSeek(
                    model=settings.llm_model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.7,
                    max_tokens=256,
                )
            except Exception as e:
                print(f"警告: LLM 初始化失败: {e}")
                self.llm = None

    def rewrite(self, query: str) -> List[str]:
        """根据配置策略改写查询"""
        if not self.llm:
            return [query]

        try:
            if self.strategy == "multi-query":
                return self._multi_query_rewrite(query)
            elif self.strategy == "hyde":
                return self._hyde_rewrite(query)
            elif self.strategy == "combined":
                multi_queries = self._multi_query_rewrite(query)
                hyde_queries = self._hyde_rewrite(query)
                combined = list(dict.fromkeys(multi_queries + hyde_queries))
                return combined[:self.num_queries]
            else:
                return [query]
        except Exception as e:
            print(f"警告: 查询重写失败: {e}")
            return [query]

    def _multi_query_rewrite(self, query: str) -> List[str]:
        """生成多个查询变体，捕捉不同角度"""
        prompt = f"""你是一个查询优化专家。请为以下用户问题生成 {self.num_queries} 个不同的查询变体，
每个变体从不同角度或不同表述方式表达同一个问题。

要求：
1. **多样化角度**：
   - 使用同义词和相关术语
   - 从不同方面提问（如：定义、功能、应用、对比等）
   - 包含专业术语和通俗表达
2. **保持核心意图**：确保所有变体都围绕原始问题的核心
3. **提高召回率**：生成的查询应该能够匹配不同的文档表述方式
4. **简洁明确**：每个查询应该简洁明了，避免过长

原始问题：{query}

输出格式（每行一个查询，不要编号和解释）：
"""

        response = self.llm.invoke(prompt)
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        queries = queries[:self.num_queries]
        if not queries:
            return [query]
        return queries

    def _hyde_rewrite(self, query: str) -> List[str]:
        """使用 HyDE 策略：先让 LLM 生成假设性答案，再基于答案改写查询"""
        prompt = f"""假设你是一个学术论文专家。请根据以下用户问题，生成一个可能的回答。
这个回答是用于改进检索的假设性答案，应该包含可能的关键词、概念、方法和结果。

用户问题：{query}

假设性回答（包含可能的答案片段、关键词、方法和术语）：
"""

        response = self.llm.invoke(prompt)
        hypothetical_answer = response.content

        refine_prompt = f"""基于以下假设性回答，改写原始问题，生成能更好检索相关文档的查询。
重点关注假设性回答中的关键信息、专业术语和具体方法。

原始问题：{query}

假设性回答：
{hypothetical_answer}

改写后的查询（突出关键概念、术语和方法）：
"""

        refined = self.llm.invoke(refine_prompt)
        return [refined.content.strip(), query]

    def rewrite_single(self, query: str) -> str:
        """单次改写，返回最优查询"""
        queries = self.rewrite(query)
        return queries[0] if queries else query
