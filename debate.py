"""
ETF 智能投资分析系统 - 仲裁引擎
"""
import re
import json
from openai import OpenAI

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_ENABLED, DEBATE_ENABLED,
    DISAGREEMENT_SCORE_THRESHOLD, DISAGREEMENT_RATING_GAP, RATING_ORDER,
    LLM_CALL_COUNT
)
from models import AgentReport


ARBITRATION_PROMPT = """你作为投资委员会仲裁员，需要裁决两位分析师的分歧。

分歧主题：{topic}

分析师A ({agent_a}) 的观点：
评级：{a_rating}
评分：{a_score}
分析：{a_analysis}
关键因子：{a_factors}
风险提示：{a_risks}

分析师B ({agent_b}) 的观点：
评级：{b_rating}
评分：{b_score}
分析：{b_analysis}
关键因子：{b_factors}
风险提示：{b_risks}

请裁决：
1. 哪位的分析更符合当前市场数据？为什么？
2. 另一位分析的缺陷是什么？
3. 综合来看，应该给谁的判断更高的权重？
4. 评分调整：应向上调整还是向下调整，调整幅度多少分（-20到+20）？

请严格按JSON格式输出：
{{
    "winner": "A或B或折中",
    "reasoning": "你的裁决理由",
    "a_flaw": "A分析的缺陷",
    "b_flaw": "B分析的缺陷",
    "score_adjustment_for_winner": +5
}}"""


# ====================== 【仲裁引擎】 ======================
class DebateEngine:
    """矛盾检测 + 选择性仲裁（LLM驱动）"""

    @staticmethod
    def _call_llm(system_prompt: str, user_prompt: str) -> str | None:
        if LLM_API_KEY == "your-api-key" or not LLM_ENABLED:
            return None
        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            LLM_CALL_COUNT["deep"] += 1
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ DebateEngine LLM调用失败: {e}")
            return None

    @staticmethod
    def detect_disagreements(reports: list[AgentReport]) -> list[dict]:
        """检测Agent间是否存在显著分歧"""
        disagreements = []
        agent_names = [r.agent_name for r in reports]
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a1, a2 = reports[i], reports[j]
                score_gap = abs(a1.score - a2.score)
                rating_gap = abs(RATING_ORDER.index(a1.rating) - RATING_ORDER.index(a2.rating))
                if score_gap >= DISAGREEMENT_SCORE_THRESHOLD or rating_gap >= DISAGREEMENT_RATING_GAP:
                    disagreements.append({
                        "agent_a": a1.agent_name, "agent_b": a2.agent_name,
                        "rating_a": a1.rating, "rating_b": a2.rating,
                        "score_a": a1.score, "score_b": a2.score,
                        "score_gap": score_gap,
                        "focus": f"{a1.agent_name}({a1.rating}) vs {a2.agent_name}({a2.rating}) 存在分歧"
                    })
        return disagreements

    @staticmethod
    def hold_debate(disagreements: list[dict], reports: list[AgentReport],
                    etf_name: str, etf_code: str) -> list[dict]:
        """执行选择性仲裁：对分歧双方进行LLM驱动的裁决"""
        if not DEBATE_ENABLED or not disagreements:
            return []
        arbitration_results = []
        reports_dict = {r.agent_name: r for r in reports}
        disagreements.sort(key=lambda d: d["score_gap"], reverse=True)
        for d in disagreements[:3]:
            agent_a = reports_dict.get(d["agent_a"])
            agent_b = reports_dict.get(d["agent_b"])
            if not agent_a or not agent_b:
                continue

            if not (LLM_ENABLED and LLM_API_KEY != "your-api-key"):
                # LLM不可用，跳过这个分歧
                continue

            topic = f"{d['agent_a']}({d['rating_a']}) vs {d['agent_b']}({d['rating_b']})" + \
                    f" 关于 {etf_name}({etf_code})"
            # 预切片分析文本（str.format不支持切片语法）
            a_analysis_300 = (agent_a.analysis or "无")[:300]
            b_analysis_300 = (agent_b.analysis or "无")[:300]
            user_prompt = ARBITRATION_PROMPT.format(
                topic=topic,
                agent_a=d["agent_a"], agent_b=d["agent_b"],
                a_rating=d["rating_a"], b_rating=d["rating_b"],
                a_score=d["score_a"], b_score=d["score_b"],
                a_analysis=a_analysis_300,
                b_analysis=b_analysis_300,
                a_factors="; ".join(agent_a.key_factors) if agent_a.key_factors else "无",
                b_factors="; ".join(agent_b.key_factors) if agent_b.key_factors else "无",
                a_risks="; ".join(agent_a.risk_warnings) if agent_a.risk_warnings else "无",
                b_risks="; ".join(agent_b.risk_warnings) if agent_b.risk_warnings else "无",
            )

            sys_prompt = f"你是ETF投资委员会仲裁员，对{etf_name}({etf_code})的分歧进行专业裁决。"
            llm_text = DebateEngine._call_llm(sys_prompt, user_prompt)
            if not llm_text:
                # LLM失败则跳过该分歧
                continue

            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', llm_text.strip())
            try:
                data = json.loads(cleaned)
            except Exception:
                # JSON解析失败则跳过
                continue

            result = {
                "topic": d["focus"],
                "agent_a": d["agent_a"],
                "agent_b": d["agent_b"],
                "a_rating": d["rating_a"],
                "b_rating": d["rating_b"],
                "a_score": d["score_a"],
                "b_score": d["score_b"],
                "winner": data.get("winner", "折中"),
                "reasoning": data.get("reasoning", ""),
                "score_adjustment": data.get("score_adjustment_for_winner", 0),
            }
            arbitration_results.append(result)

        return arbitration_results
