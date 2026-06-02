"""
ETF 智能投资分析系统 - LLM多智能体基类
"""
import numpy as np
import re
import json
from datetime import datetime
from openai import OpenAI

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_ENABLED, RATING_ORDER, REQUEST_DELAY,
    CACHE_OPINION,
    QUICK_LLM_API_KEY, QUICK_LLM_BASE_URL, QUICK_LLM_MODEL,
    QUICK_LLM_MAX_TOKENS, QUICK_LLM_TEMPERATURE,
    LLM_CALL_COUNT
)
from models import AgentReport, LLMOutput
from memory import MemoryRetriever


# ====================== 【LLM多智能体基类】 ======================
class BaseLLMAgent:
    """所有LLM驱动Agent的基类。LLM失败时自动fallback到规则评分。"""

    ROLE_NAME = "基础分析师"
    SYSTEM_PROMPT = "你是一个专业的金融分析师。请基于提供的数据进行分析。"
    # 各Agent可覆盖以下配置实现temperature/model多样性
    AGENT_TEMPERATURE = None  # None=使用LLM_TEMPERATURE全局值
    AGENT_MODEL = None        # None=使用LLM_MODEL全局值
    AGENT_USE_QUICK_MODEL = False  # True=使用快速模型（轻量分析）

    # 记忆系统配置
    MEMORY_ENABLED = True   # 全局开关：注入历史分析记录
    MEMORY_DAYS = 20        # 检索最近N天的历史记录

    def __init__(self):
        api_key = LLM_API_KEY if not self.AGENT_USE_QUICK_MODEL else QUICK_LLM_API_KEY
        base_url = LLM_BASE_URL if not self.AGENT_USE_QUICK_MODEL else QUICK_LLM_BASE_URL
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

        if self.AGENT_USE_QUICK_MODEL:
            self.quick_client = self.client
        else:
            self.quick_client = OpenAI(
                api_key=QUICK_LLM_API_KEY, base_url=QUICK_LLM_BASE_URL
            ) if QUICK_LLM_API_KEY else None

    def _enrich_with_memory(self, etf_code: str, data_text: str) -> str:
        """
        将历史分析记忆注入到数据文本中。
        全局/市场级Agent（MACRO/MONETARY/POLICY）跳过记忆注入。
        """
        if not self.MEMORY_ENABLED or etf_code in ("MACRO", "MONETARY", "POLICY"):
            return data_text
        try:
            memory = MemoryRetriever.retrieve(etf_code, days=self.MEMORY_DAYS)
            if memory:
                return memory + "\n\n" + data_text
        except Exception:
            pass
        return data_text

    def _build_user_prompt(self, etf_name: str, etf_code: str, data_text: str) -> str:
        return f"""标的：{etf_name}（{etf_code}）
时间：{datetime.now().strftime('%Y-%m-%d')}

数据：
{data_text}

请严格按照以下JSON格式输出（不要markdown代码块标记）：
{{
    "rating": "看多/看空/中性/强烈看多/强烈看空",
    "score": 0-100的整数分数,
    "analysis": "你的分析报告（200-400字）",
    "key_factors": ["关键因子1", "关键因子2"],
    "risk_warnings": ["风险点1", "风险点2"],
    "confidence": 0-1之间的置信度
}}"""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> LLMOutput | None:
        """调用 LLM 并返回 Pydantic 验证后的结构化输出。失败返回 None。"""
        if not LLM_ENABLED:
            return None

        use_quick = self.AGENT_USE_QUICK_MODEL
        client = self.quick_client if use_quick else self.client
        if client is None:
            print(f"  ⚠️ {self.ROLE_NAME} LLM客户端未配置（缺少API Key）")
            return None

        try:
            route = "quick" if use_quick else "deep"
            temp = self.AGENT_TEMPERATURE if self.AGENT_TEMPERATURE is not None else (
                QUICK_LLM_TEMPERATURE if use_quick else LLM_TEMPERATURE
            )
            max_tok = QUICK_LLM_MAX_TOKENS if use_quick else LLM_MAX_TOKENS
            model = self.AGENT_MODEL if self.AGENT_MODEL is not None else (
                QUICK_LLM_MODEL if use_quick else LLM_MODEL
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            text = resp.choices[0].message.content.strip()
            # 清理可能的markdown代码块
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            # 统计调用次数
            LLM_CALL_COUNT["total"] += 1
            LLM_CALL_COUNT[route] += 1

            # 通过 Pydantic 模型验证
            raw = json.loads(text)
            validated = LLMOutput.from_llm_json(raw)
            if validated is None:
                print(f"  ⚠️ {self.ROLE_NAME} LLM输出格式无效，使用规则评分")
            return validated
        except json.JSONDecodeError as e:
            print(f"  ⚠️ {self.ROLE_NAME} LLM返回非JSON: {e}")
            return None
        except Exception as e:
            print(f"  ⚠️ {self.ROLE_NAME} LLM调用失败 ({route}): {e}")
            return None

    def _parse_to_report(self, etf_code: str, etf_name: str,
                         llm_output: LLMOutput | None, fallback_score: float,
                         fallback_rating: str | None = None) -> AgentReport:
        if llm_output is None:
            return AgentReport(
                agent_name=self.ROLE_NAME,
                etf_code=etf_code,
                etf_name=etf_name,
                rating=fallback_rating or self._score_to_rating(fallback_score),
                score=round(fallback_score, 1),
                analysis=f"【规则评分模式】LLM不可用，基于规则模型评分 {fallback_score} 分。",
                key_factors=["规则评分（LLM fallback）"],
                risk_warnings=[],
                confidence=0.5,
                source="rule_fallback"
            )

        return AgentReport(
            agent_name=self.ROLE_NAME,
            etf_code=etf_code,
            etf_name=etf_name,
            rating=llm_output.rating if llm_output.rating in RATING_ORDER else "中性",
            score=float(np.clip(llm_output.score, 0, 100)),
            analysis=llm_output.analysis,
            key_factors=llm_output.key_factors,
            risk_warnings=llm_output.risk_warnings,
            confidence=float(np.clip(llm_output.confidence, 0, 1)),
            source="llm"
        )

    @staticmethod
    def _score_to_rating(score: float) -> str:
        if score >= 80: return "强烈看多"
        if score >= 65: return "看多"
        if score >= 45: return "中性"
        if score >= 30: return "看空"
        return "强烈看空"
