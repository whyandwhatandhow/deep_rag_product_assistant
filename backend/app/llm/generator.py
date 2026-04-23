# backend/app/llm/generator.py
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from app.core.config import settings
from .prompts import GROUNDED_PROMPT, LOOSE_GROUNDED_PROMPT

# 强制加载 .env（确保在 Generator 初始化前生效）
load_dotenv(override=True)

class Generator:
    def __init__(self):
        # 优先从 settings / 环境变量获取
        api_key = settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = settings.deepseek_base_url or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

        self.llm = None
        if api_key and api_key != "your_api_key_here":
            try:
                self.llm = ChatDeepSeek(
                    model=settings.llm_model,          # deepseek-chat
                    api_key=api_key,
                    base_url=base_url,                 # 明确传入，避免默认 base 校验
                    temperature=0.0,                   # 深 RAG 必须低温度
                    max_tokens=1024,
                )
                print(f"Generator 初始化完成 | 模型: {settings.llm_model} | base_url: {base_url}")
            except Exception as e:
                print(f"警告: LLM 初始化失败: {e}")
                self.llm = None
        else:
            print("警告: 未配置有效的 DEEPSEEK_API_KEY，将使用回退模式")
            self.llm = None

    def generate(self, question: str, context: str) -> str:
        """严格 grounded 生成"""
        if not self.llm:
            # 回退模式：基于规则的简单回答
            if context:
                return f"基于检索到的信息：{context[:200]}...（由于 API 密钥未配置，使用回退模式）"
            else:
                return "知识库中未找到依据（由于 API 密钥未配置，使用回退模式）"

        try:
            if len(context) > 100:
                prompt = GROUNDED_PROMPT.format(question=question, context=context)
            else:
                prompt = LOOSE_GROUNDED_PROMPT.format(question=question, context=context)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"警告: 生成失败: {e}")
            # 生成失败时的回退
            if context:
                return f"基于检索到的信息：{context[:200]}...（生成失败，使用回退模式）"
            else:
                return "知识库中未找到依据（生成失败，使用回退模式）"
