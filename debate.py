"""
ETF 智能投资分析系统 - 辩论引擎
"""
import re
import json
from openai import OpenAI

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_ENABLED, DEBATE_ENABLED, DEBATE_ROUNDS,
    DISAGREEMENT_SCORE_THRESHOLD, DISAGREEMENT_RATING_GAP, RATING_ORDER
)
from models import AgentReport


# ====================== 【辩论引擎】 ======================
class DebateEngine:
    """矛盾检测 + 选择性辩论（LLM驱动）"""

    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        ) if LLM_API_KEY != "your-api-key" else None

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
        """执行选择性辩论：对分歧双方进行真实LLM驱动辩论"""
        if not DEBATE_ENABLED or not disagreements:
            return []
        debate_logs = []
        reports_dict = {r.agent_name: r for r in reports}
        for d in disagreements[:3]:
            agent_a = reports_dict.get(d["agent_a"])
            agent_b = reports_dict.get(d["agent_b"])
            if not agent_a or not agent_b:
                continue
            debate_log = {"topic": d["focus"], "agents": [d["agent_a"], d["agent_b"]], "rounds": []}

            if LLM_ENABLED and LLM_API_KEY != "your-api-key":
                sys_prompt = f"""你是ETF投研辩论主持人。以下两位分析师对{etf_name}({etf_code})存在分歧。
请基于他们的实际分析内容，生成真实的辩论对话。保持专业、数据驱动的风格，每个发言控制在150字以内。"""

                r1_prompt = f"""分析师A（{d['agent_a']}）评级{d['rating_a']}({d['score_a']}分)，完整分析：{agent_a.analysis}

分析师B（{d['agent_b']}）评级{d['rating_b']}({d['score_b']}分)，完整分析：{agent_b.analysis}

请生成第1轮辩论：
1. {d['agent_a']}陈述自己的核心观点和论据
2. {d['agent_b']}针对{d['agent_a']}的观点进行反驳，引用自己的分析数据

以JSON格式输出（不要markdown代码块标记）：
{{"statement": "A的陈述", "rebuttal": "B的反驳"}}"""

                r1_text = DebateEngine._call_llm(sys_prompt, r1_prompt)
                r1_data = None
                if r1_text:
                    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', r1_text.strip())
                    try:
                        r1_data = json.loads(cleaned)
                    except Exception:
                        pass

                if r1_data and "statement" in r1_data and "rebuttal" in r1_data:
                    debate_log["rounds"].append({
                        "round": 1,
                        "statement": f"{d['agent_a']}: {r1_data['statement']}",
                        "rebuttal": f"{d['agent_b']}: {r1_data['rebuttal']}"
                    })
                else:
                    debate_log["rounds"].append({
                        "round": 1,
                        "statement": f"{d['agent_a']}: 评级{d['rating_a']}({d['score_a']}分)。{agent_a.analysis[:200]}",
                        "rebuttal": f"{d['agent_b']}: 评级{d['rating_b']}({d['score_b']}分)。{agent_b.analysis[:200]}"
                    })

                if DEBATE_ROUNDS >= 2:
                    r2_prompt = f"""基于第1轮辩论：
A: {debate_log['rounds'][0]['statement']}
B: {debate_log['rounds'][0]['rebuttal']}

分析师A（{d['agent_a']}）关键因子：{', '.join(agent_a.key_factors)}
分析师B（{d['agent_b']}）关键因子：{', '.join(agent_b.key_factors)}

请生成第2轮辩论：
1. {d['agent_b']}针对{d['agent_a']}的论据提出质疑和反问
2. {d['agent_a']}回应{d['agent_b']}的质疑，维护自己的观点

以JSON格式输出（不要markdown代码块标记）：
{{"challenge": "B的质疑", "response": "A的回应"}}"""

                    r2_text = DebateEngine._call_llm(sys_prompt, r2_prompt)
                    r2_data = None
                    if r2_text:
                        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', r2_text.strip())
                        try:
                            r2_data = json.loads(cleaned)
                        except Exception:
                            pass

                    if r2_data and "challenge" in r2_data and "response" in r2_data:
                        debate_log["rounds"].append({
                            "round": 2,
                            "challenge": f"{d['agent_b']}: {r2_data['challenge']}",
                            "response": f"{d['agent_a']}: {r2_data['response']}"
                        })
                    else:
                        key_b = ', '.join(agent_b.key_factors[:2]) if agent_b.key_factors else '潜在风险'
                        key_a = ', '.join(agent_a.key_factors[:3]) if agent_a.key_factors else '多维度数据'
                        debate_log["rounds"].append({
                            "round": 2,
                            "challenge": f"{d['agent_b']}反问：{d['agent_a']}的{d['rating_a']}判断是否低估了{key_b}等因素的影响？",
                            "response": f"{d['agent_a']}回应：{key_a}支撑我的判断，{d['agent_b']}提示的风险已标注在报告的风险提示中。"
                        })
            else:
                key_b = ', '.join(agent_b.key_factors[:2]) if agent_b.key_factors else '潜在风险'
                key_a = ', '.join(agent_a.key_factors[:3]) if agent_a.key_factors else '多维度数据'
                debate_log["rounds"].append({
                    "round": 1,
                    "statement": f"{d['agent_a']}: 评级{d['rating_a']}({d['score_a']}分)。核心逻辑：{agent_a.analysis[:200]}",
                    "rebuttal": f"{d['agent_b']}: 评级{d['rating_b']}({d['score_b']}分)。核心逻辑：{agent_b.analysis[:200]}"
                })
                if DEBATE_ROUNDS >= 2:
                    debate_log["rounds"].append({
                        "round": 2,
                        "challenge": f"{d['agent_b']}反问：{d['agent_a']}的{d['rating_a']}判断是否忽略了{key_b}？",
                        "response": f"{d['agent_a']}回应：{key_a}支持我的判断，{d['agent_b']}提醒的风险点已记录。"
                    })
            debate_logs.append(debate_log)
        return debate_logs
