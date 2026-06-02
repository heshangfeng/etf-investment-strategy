"""
ETF 智能投资分析系统 - LLM多智能体层
"""
import numpy as np
import pandas as pd
import re
import json
import os
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import akshare as ak

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_ENABLED, RATING_ORDER, REQUEST_DELAY,
    CACHE_OPINION,
    QUICK_LLM_API_KEY, QUICK_LLM_BASE_URL, QUICK_LLM_MODEL,
    QUICK_LLM_MAX_TOKENS, QUICK_LLM_TEMPERATURE,
    LLM_CALL_COUNT
)
from models import AgentReport
from keywords import INDUSTRY_POS
from data import DataCollectAgent, PublicOpinionAgent


# ====================== 【LLM多智能体基类】 ======================
class BaseLLMAgent:
    """所有LLM驱动Agent的基类。LLM失败时自动fallback到规则评分。"""

    ROLE_NAME = "基础分析师"
    SYSTEM_PROMPT = "你是一个专业的金融分析师。请基于提供的数据进行分析。"
    # 各Agent可覆盖以下配置实现temperature/model多样性
    AGENT_TEMPERATURE = None  # None=使用LLM_TEMPERATURE全局值
    AGENT_MODEL = None        # None=使用LLM_MODEL全局值
    AGENT_USE_QUICK_MODEL = False  # True=使用快速模型（轻量分析）

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

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict | None:
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

            return json.loads(text)
        except Exception as e:
            print(f"  ⚠️ {self.ROLE_NAME} LLM调用失败 ({route if 'route' in dir() else 'unknown'}): {e}")
            return None

    def _parse_to_report(self, etf_code: str, etf_name: str,
                         llm_output: dict | None, fallback_score: float,
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

        rating = llm_output.get("rating", "中性")
        if rating not in RATING_ORDER:
            rating = "中性"

        return AgentReport(
            agent_name=self.ROLE_NAME,
            etf_code=etf_code,
            etf_name=etf_name,
            rating=rating,
            score=float(np.clip(llm_output.get("score", 50), 0, 100)),
            analysis=llm_output.get("analysis", ""),
            key_factors=llm_output.get("key_factors", []),
            risk_warnings=llm_output.get("risk_warnings", []),
            confidence=float(np.clip(llm_output.get("confidence", 0.5), 0, 1)),
            source="llm"
        )

    @staticmethod
    def _score_to_rating(score: float) -> str:
        if score >= 80: return "强烈看多"
        if score >= 65: return "看多"
        if score >= 45: return "中性"
        if score >= 30: return "看空"
        return "强烈看空"


# ====================== 【LLM多智能体 - 宏观分析】 ======================
class MacroAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "宏观分析智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是拥有20年经验的央行宏观分析师。
你的分析框架：
1. 市场成交量代表流动性和参与度——量能决定行情级别
2. 成交额>10000亿=强趋势市场，7000-10000亿=震荡，<7000亿=弱势
3. 给出基于流动性的宏观环境和仓位建议
务必简洁专业，数据驱动。"""

    def run(self) -> AgentReport:
        vol = DataCollectAgent.get_market_total_volume()
        data_text = f"今日全市场成交额：{vol:.0f}亿元"

        # LLM分析
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("全市场", "MACRO", data_text))

        # Fallback: 基于阈值的规则
        if llm_out is None:
            if vol >= 10000:
                score, rating, analysis = 90, "强烈看多", "强趋势行情"
            elif vol >= 7000:
                score, rating, analysis = 60, "看多", "震荡行情"
            else:
                score, rating, analysis = 25, "看空", "弱势行情"
            return AgentReport(
                agent_name=self.ROLE_NAME, etf_code="MACRO", etf_name="全市场",
                rating=rating, score=score,
                analysis=f"【规则评分】{analysis} | 成交额{vol:.0f}亿",
                key_factors=[f"成交额{vol:.0f}亿"],
                risk_warnings=["规则评分（LLM不可用）"],
                confidence=0.6, source="rule_fallback",
                data_summary={"market_volume": vol}
            )

        return self._parse_to_report("MACRO", "全市场", llm_out,
                                      fallback_score=50, fallback_rating="中性")


# ====================== 【LLM多智能体 - 价值估值】 ======================
class ValueAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "价值估值智能体"
    AGENT_TEMPERATURE = 0.3
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是信奉格雷厄姆-巴菲特价值投资理念的分析师。
分析框架：
1. PE/PB百分位是核心——百分位<30%为低估，>70%为高估
2. 低估值+合理百分位=安全边际充足，看好
3. 高估值+高百分位=泡沫风险，看空
4. 百分位在30%-70%之间为估值合理区域
给出基于估值安全边际的判断。"""

    def run(self, etf_code: str, etf_name: str, index_code: str) -> AgentReport:
        d = DataCollectAgent.get_index_val(index_code)
        score = round(100 - d["pe_percent"], 2)
        data_text = f"PE(TTM): {d['pe']} | PB: {d['pb']} | PE近5年百分位: {d['pe_percent']}%"
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 技术趋势】 ======================
class TechAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "技术趋势智能体"
    AGENT_TEMPERATURE = 0.3
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是拥有15年经验的技术分析师，擅长趋势识别。
分析框架：
1. 价格与均线关系：价格在MA5和MA20之上=多头排列，之下=空头排列
2. 均线金叉(MA5上穿MA20)=看多信号，死叉=看空信号
3. 近期波动率异常高=风险加大
4. 连续上涨/下跌天数反映短期动能
结合价格位置、均线形态、波动率给出判断。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 20:
            return AgentReport(self.ROLE_NAME, etf_code, etf_name, "中性", 50,
                               "数据不足20个交易日", ["数据不足"], [], 0.3, {}, "rule_fallback")
        close, m5, m20 = df["close"].iloc[-1], df["ma5"].iloc[-1], df["ma20"].iloc[-1]
        ts = 50
        if close > m5 and close > m20: ts += 18
        if m5 > m20: ts += 12
        if close < m5 and close < m20: ts -= 25
        score = float(np.clip(ts, 0, 100))
        recent_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        data_text = (f"最新价: {close:.3f} | MA5: {m5:.3f} | MA20: {m20:.3f}\n"
                     f"均线关系: {'多头' if m5 > m20 else '空头'}排列\n"
                     f"近5日涨跌: {recent_5d:.2f}%\n"
                     f"近5日平均波动率: {df['volatility'].tail(5).mean():.4f}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 舆情情绪】 ======================
class SentimentAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "舆情情绪智能体"
    AGENT_TEMPERATURE = 0.5
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是行为金融学专家，擅长识别市场情绪。
分析框架：
1. 情绪分数>70为乐观（可能过度乐观），<30为恐慌（可能过度悲观）
2. 舆情趋势比单日分数更重要——持续回暖或持续走弱是强烈信号
3. 结合行业关键词判断是否存在实质性利好/利空
4. 关注多空交织情况——矛盾信号意味着市场分歧大
重点关注情绪极端值和趋势变化。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        cache_key = f"{etf_code}_{etf_name}"
        if cache_key in CACHE_OPINION:
            op = CACHE_OPINION[cache_key]
        else:
            op = PublicOpinionAgent.llm_run(etf_name, etf_code)
        data_text = (f"舆情情绪分数: {op['opinion_score']}/100\n"
                     f"情绪标签: {op['opinion_tag']}\n"
                     f"舆情趋势: {op['opinion_trend']}\n"
                     f"情绪关键词: {op['keywords']}\n"
                     f"资讯摘要: {op['news_content'][:300]}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        score = op['opinion_score']
        if score >= 70: fr = "强烈看多"
        elif score >= 60: fr = "看多"
        elif score >= 40: fr = "中性"
        elif score >= 30: fr = "看空"
        else: fr = "强烈看空"
        report = self._parse_to_report(etf_code, etf_name, llm_out, score, fr)
        report.data_summary = {"opinion_score": op['opinion_score'], "opinion_trend": op['opinion_trend'],
                               "keywords": op['keywords'], "warn_alert": op['warn_alert']}
        return report


# ====================== 【LLM多智能体 - 资金流向】 ======================
class FundFlowAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "资金流向智能体"
    AGENT_TEMPERATURE = 0.5
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是专注于资金流分析的市场老手。
分析框架：
1. 北向资金持续流入=外资看好，是重要正向信号
2. 成交量放大(量比>1.2)=资金活跃参与，缩量=观望
3. 折溢价率>1.5%=溢价过高有回落风险
4. 量价配合关系：放量上涨健康，放量下跌危险
结合北向、量比、折溢价综合判断资金面。"""

    def run(self, etf_code: str, etf_name: str, index_code: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(10).mean()
        north = DataCollectAgent.get_north_flow(index_code)
        premium = DataCollectAgent.get_etf_premium(etf_code)
        s1 = np.clip(50 + (vol_ratio - 1) * 30, 0, 100)
        s2 = np.clip(50 + north / 100_000_000 * 5, 0, 100)
        score = round((s1 + s2) / 2, 2)
        # 公募股票仓位
        fund_pos_info = ""
        try:
            fp = ak.fund_stock_position_lg()
            if fp is not None and len(fp) > 0:
                fp_val = float(fp.iloc[-1]["股票仓位"])
                fund_pos_info = f"\n公募股票仓位: {fp_val}%"
        except:
            pass

        data_text = (f"成交量比(近10日均值): {vol_ratio:.2f}\n"
                     f"北向近5日净流入: {north/100000000:.2f}亿\n"
                     f"折溢价率: {premium*100:.3f}%\n"
                     f"资金分项(s1量比): {s1:.1f} | (s2北向): {s2:.1f}"
                     f"{fund_pos_info}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 风险管理】 ======================
class RiskManagerAgent(BaseLLMAgent):
    ROLE_NAME = "风险管理智能体"
    AGENT_TEMPERATURE = 0.15
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是偏保守的首席风控官，对下行风险极度敏感。
分析框架：
1. 折溢价>1.5%=溢价过高，存在回落风险（扣分）
2. 日波动率>3%=波动剧烈，不适合稳健仓位（扣分）
3. 成交量<5000万=流动性不足，买卖价差大（扣分）
4. 近期最大回撤幅度越大=风险越高
风控视角：宁可错过，不可做错。高评分=低风险。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        premium = DataCollectAgent.get_etf_premium(etf_code)
        vol = df["volatility"].iloc[-1]
        liq = df["volume"].iloc[-1]
        score = 100.0
        risk_detail = []
        if premium > 0.015: score -= 30; risk_detail.append(f"溢价过高({premium*100:.2f}%)")
        if vol > 0.03: score -= 15; risk_detail.append(f"波动率高({vol:.4f})")
        if liq < 5000: score -= 25; risk_detail.append(f"流动性不足(成交量{liq:.0f})")
        score = float(np.clip(score, 0, 100))

        # 两融杠杆风险
        margin_note = ""
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_m = margin['szse_margin'] + margin['sse_margin']
            if total_m > 20000: score -= 10; risk_detail.append("两融余额过高，市场杠杆风险大")
            margin_note = f"两融余额: {total_m:.0f}亿"
        except:
            pass

        # VIX情绪指标
        vix = DataCollectAgent.get_ivix(etf_code)
        vix_note = f"隐含波动率(VIX): {vix:.1f}"
        if vix > 30:
            score -= 10
            risk_detail.append(f"VIX恐慌({vix:.1f}>30)")
        elif vix < 15:
            score -= 5
            risk_detail.append(f"VIX过低({vix:.1f}<15), 市场可能自满")

        data_text = (f"折溢价率: {premium*100:.3f}%\n"
                     f"日波动率: {vol:.4f}\n"
                     f"成交量: {liq:.0f}\n"
                     f"{vix_note}\n"
                     f"{margin_note}\n"
                     f"风险扣分原因: {'; '.join(risk_detail) if risk_detail else '无明显风险'}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 行业纵析】 ======================
class IndustryAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "行业纵析智能体"
    AGENT_TEMPERATURE = 0.4
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是深耕行业的资深研究员，精通行业轮动分析。
分析框架：
1. 识别ETF所属行业赛道（从名称判断）
2. 分析行业所处周期位置
3. 匹配行业专属关键词判断当前景气度
4. 相对强度——该ETF近期相对大盘的表现强弱
5. 行业动量——该ETF近期的涨幅趋势

A股行业轮动特征：
- 强势行业持续期通常3-6个月
- 资金会从高位行业流向低位行业（均值回归）
- 政策催化可以提前启动行业轮动
给出基于行业基本面和轮动位置的独立判断。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        pos_words, neg_words = PublicOpinionAgent.get_industry_keywords(etf_name)
        matched_industries = [ind for ind in INDUSTRY_POS if ind in etf_name]
        industry = matched_industries[0] if matched_industries else "通用"

        df = DataCollectAgent.get_etf_price(etf_code)
        close = df["close"].iloc[-1]

        ret_20d = (close / df["close"].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        ret_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0

        # 相对强度：对比沪深300
        relative_strength = 0.0
        strength_desc = "无法计算相对强度"
        try:
            benchmark = DataCollectAgent.get_etf_price("510300")
            bench_close = benchmark["close"].iloc[-1]
            bench_ret_20d = (bench_close / benchmark["close"].iloc[-20] - 1) * 100 if len(benchmark) >= 20 else 0
            relative_strength = ret_20d - bench_ret_20d
            strength_desc = f"跑赢大盘{relative_strength:.1f}%" if relative_strength > 0 else f"跑输大盘{abs(relative_strength):.1f}%"
        except Exception:
            pass

        # 板块资金流向
        sector_flow_info = ""
        try:
            sector_flow = DataCollectAgent.get_sector_fund_flow()
            if industry in sector_flow:
                sf = sector_flow[industry]
                sector_flow_info = f"板块资金净流入: {sf['流入']/1e8:.1f}亿 | 排名: {sf['流入排名']}"
            else:
                sector_flow_info = "暂无该行业板块资金流向数据"
        except:
            pass

        data_text = (f"所属行业: {industry}\n"
                     f"行业正向词: {', '.join(pos_words)}\n"
                     f"行业负向词: {', '.join(neg_words)}\n"
                     f"近5日涨幅: {ret_5d:.2f}%\n"
                     f"近20日涨幅: {ret_20d:.2f}%\n"
                     f"相对强度(20日vs沪深300): {strength_desc}\n"
                     f"{sector_flow_info}")

        # 规则评分：行业归属+动量贡献
        score = 50.0
        if industry != "通用": score += 10
        score += float(np.clip(ret_20d * 0.5, -20, 20))
        score = float(np.clip(score, 0, 100))

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 货币政策/流动性】 ======================
class MonetaryPolicyAgent(BaseLLMAgent):
    ROLE_NAME = "货币政策智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是前央行研究员，专注于货币政策与流动性分析。
分析框架：
1. 政策利率（MLF、LPR、7天逆回购利率）变动趋势
2. 存款准备金率(RRR)水平
3. 社融/M2增速
4. 银行间质押式回购利率(DR007)
5. 人民币汇率(USDCNY)对宽松空间的制约
请基于可获得的数据给出流动性环境判断。"""

    def run(self) -> AgentReport:
        try:
            df_lpr = ak.interest_rate_lpr()
            lpr_1y = float(df_lpr[df_lpr["期限种类"] == "1年期"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
            lpr_5y = float(df_lpr[df_lpr["期限种类"] == "5年期以上"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
        except Exception:
            lpr_1y, lpr_5y = None, None

        try:
            fx = ak.spot_quote()
            usdcny = None
            if fx is not None:
                usd_row = fx[fx["名称"].str.contains("美元", na=False)]
                if len(usd_row) > 0:
                    usdcny = float(usd_row["现价"].iloc[0])
        except Exception:
            usdcny = None

        score = 50.0
        signals = []

        if lpr_1y is not None:
            signals.append(f"1年期LPR: {lpr_1y}%")
            if lpr_1y <= 3.1: score += 20
            elif lpr_1y >= 3.85: score -= 15

        if lpr_5y is not None:
            signals.append(f"5年期LPR: {lpr_5y}%")

        if usdcny is not None:
            signals.append(f"USDCNY: {usdcny}")
            if usdcny > 7.3: score -= 15
            elif usdcny < 6.8: score += 10

        # 加入中美利差
        try:
            bond = DataCollectAgent.get_bond_yield()
            signals.append(f"中美利差: {bond['spread']}% (中国{bond['cn_10y']}%-美国{bond['us_10y']}%)")
        except:
            pass

        # 加入融资融券（两融情绪）
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_margin = margin['szse_margin'] + margin['sse_margin']
            signals.append(f"两融余额: {total_margin:.0f}亿 | 融券: {margin['szse_short']:.0f}亿")
            if total_margin > 15000: score += 5
        except:
            pass

        # 宏观数据：M2/SHIBOR
        try:
            m2 = ak.macro_china_m2_yearly()
            if m2 is not None and len(m2) > 0:
                m2_val = float(m2.iloc[-1]["同比增速"])
                signals.append(f"M2同比: {m2_val}%")
                if m2_val > 10: score += 5
        except:
            pass
        try:
            shibor = ak.macro_china_shibor_all()
            if shibor is not None and len(shibor) > 0:
                on_rate = float(shibor.iloc[-1]["ON"]) if "ON" in shibor.columns else 0
                signals.append(f"SHIBOR隔夜: {on_rate}%")
        except:
            pass
        try:
            pmi_df = ak.macro_china_pmi()
            if pmi_df is not None:
                pmi_val = float(pmi_df.tail(1).values[0][1])
                signals.append(f"制造业PMI: {pmi_val}")
                if pmi_val > 52: score += 10
                elif pmi_val < 48: score -= 10
        except:
            pass

        score = float(np.clip(score, 0, 100))
        data_text = "\n".join(signals) if signals else "暂无实时货币政策数据"

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国市场", "MONETARY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("MONETARY", "货币政策", llm_out, score, rating)
        report.data_summary = {"lpr_1y": lpr_1y, "lpr_5y": lpr_5y, "usdcny": usdcny}
        return report


# ====================== 【LLM多智能体 - 政策事件/政治周期】 ======================
class PolicyEventAgent(BaseLLMAgent):
    ROLE_NAME = "政策事件智能体"
    AGENT_TEMPERATURE = 0.6
    SYSTEM_PROMPT = """你是资深政策分析师，专注中国政治经济周期。
分析框架：
1. 识别当前所处的政策周期阶段
2. 重大会议窗口：两会(3月)、政治局会议(4/7/10/12月)、中央经济工作会议(12月)
3. 行业政策催化——近期是否有针对特定行业的重大政策出台
4. 政策基调判断：宽松/中性/收紧
A股特征：两会前后春季躁动，政治局会议定调影响季度级别方向。"""

    # 重要会议日历 (月, 开始日, 结束日)
    KEY_EVENTS = [
        ("两会", 3, 1, 3, 15),
        ("政治局会议(4月)", 4, 15, 4, 30),
        ("政治局会议(7月)", 7, 15, 7, 31),
        ("政治局会议(10月)", 10, 15, 10, 31),
        ("中央经济工作会议", 12, 1, 12, 15),
    ]

    def run(self) -> AgentReport:
        now = datetime.now()
        month, day = now.month, now.day

        current_event = None
        for name, sm, sd, em, ed in self.KEY_EVENTS:
            try:
                ev_start = datetime(now.year, sm, sd)
                ev_end = datetime(now.year, em, ed)
                if ev_start <= now <= ev_end:
                    current_event = f"当前处于{name}窗口期"
                    break
                if now < ev_start:
                    days_to = (ev_start - now).days
                    if days_to <= 30:
                        current_event = f"距离{name}还有{days_to}天"
                        break
            except Exception:
                continue

        season_effect = ""
        if month in (1, 2, 3): season_effect = "春季躁动窗口"
        elif month == 4: season_effect = "年报季+政治局会议定调"
        elif month in (7, 8): season_effect = "中报季+政治局会议落地"
        elif month in (10, 11): season_effect = "三季报+年末政策定调"
        elif month == 12: season_effect = "中央经济工作会议+机构调仓"

        data_text = (f"当前日期: {now.strftime('%Y-%m-%d')}\n"
                     f"季节效应: {season_effect}\n"
                     f"会议窗口: {current_event or '无重要会议窗口'}\n"
                     f"月份特征: {month}月")

        score = 55.0
        if current_event: score += 10
        if month in (3, 7, 12): score += 8
        if month in (1, 2): score += 5
        score = float(np.clip(score, 0, 100))

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国政策周期", "POLICY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("POLICY", "政策周期", llm_out, score, rating)
        report.data_summary = {"current_event": current_event, "season_effect": season_effect, "month": month}
        return report


# ====================== 【LLM多智能体 - 零售情绪结构】 ======================
class RetailSentimentAgent(BaseLLMAgent):
    ROLE_NAME = "零售情绪智能体"
    AGENT_TEMPERATURE = 0.5
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是行为金融学量化分析师，专注A股散户情绪测量。
分析框架：
1. 量比（成交量/20日均量）——放量过大=情绪过热，缩量=冷清
2. 价格位置（52周高低位置）——接近高点=亢奋，接近低点=恐慌
3. 散户情绪是典型的反向指标——极度乐观时见顶，极度悲观时见底
请基于量价数据给出情绪判断。"""

    def run(self, etf_code: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)

        vol_20_mean = df["volume"].tail(20).mean()
        vol_today = df["volume"].iloc[-1]
        turnover_ratio = vol_today / vol_20_mean if vol_20_mean > 0 else 1.0

        close = df["close"].iloc[-1]
        high_52w = df["close"].tail(250).max() if len(df) >= 250 else df["close"].max()
        low_52w = df["close"].tail(250).min() if len(df) >= 250 else df["close"].min()
        pos_from_low = (close - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50

        score = 50.0
        signals = []

        if turnover_ratio > 2.0:
            signals.append(f"放量异常(量比{turnover_ratio:.2f})"); score -= 10
        elif turnover_ratio > 1.5:
            signals.append(f"放量(量比{turnover_ratio:.2f})"); score += 5
        elif turnover_ratio < 0.5:
            signals.append(f"缩量(量比{turnover_ratio:.2f})"); score -= 5
        else:
            signals.append(f"量能正常(量比{turnover_ratio:.2f})")

        if pos_from_low > 90:
            signals.append("接近年内高点"); score -= 8
        elif pos_from_low < 10:
            signals.append("接近年内低点"); score += 8
        elif 40 <= pos_from_low <= 60:
            signals.append("价格中位区间"); score += 3

        # 加入两融数据
        margin_info = ""
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_m = margin['szse_margin'] + margin['sse_margin']
            margin_info = f"两融余额: {total_m:.0f}亿"
        except:
            pass

        # 市场活跃度数据
        activity_info = ""
        try:
            act = ak.stock_market_activity_legu()
            if act is not None and "换手率" in act.columns:
                turnover_rate = float(act["换手率"].iloc[-1])
                activity_info = f"\n全市场换手率: {turnover_rate}%"
        except:
            pass

        score = float(np.clip(score, 0, 100))
        data_text = (f"量比(20日均值): {turnover_ratio:.2f}\n"
                     f"52周价格位置: {pos_from_low:.1f}%\n"
                     f"情绪信号: {'; '.join(signals)}\n"
                     f"{margin_info}"
                     f"{activity_info}")

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("市场情绪", etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report(etf_code, "市场情绪", llm_out, score, rating)
        report.data_summary = {"turnover_ratio": turnover_ratio, "pos_from_low": pos_from_low}
        return report


# ====================== 【LLM多智能体 - 跨资产/跨市场联动】 ======================
class CrossMarketAgent(BaseLLMAgent):
    ROLE_NAME = "跨市场联动智能体"
    AGENT_TEMPERATURE = 0.6
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是全球宏观策略分析师，专注跨市场信号传导。
分析框架：
1. 人民币汇率(USDCNY)——升值利好A股，贬值承压
2. 美债收益率——全球资产定价锚
3. 美元指数——美元强弱影响外资流向
4. 黄金价格——避险情绪指标
传导逻辑：美债利率+美元同时走强=新兴市场承压；人民币升值+美元弱=利好A股。"""

    def run(self, etf_name: str, etf_code: str) -> AgentReport:
        signals = []
        usdcny = None
        try:
            fx = ak.spot_quote()
            cny_row = fx[fx["名称"].str.contains("美元", na=False)]
            if len(cny_row) > 0:
                usdcny = float(cny_row["现价"].iloc[0])
                signals.append(f"USDCNY: {usdcny}")
        except Exception:
            pass

        # 加入中美利差数据
        try:
            bond = DataCollectAgent.get_bond_yield()
            signals.append(f"中国10Y国债: {bond['cn_10y']}% | 美国10Y国债: {bond['us_10y']}% | 中美利差: {bond['spread']}%")
        except:
            pass

        score = 50.0
        if usdcny is not None:
            if usdcny > 7.3: score -= 15
            elif usdcny < 6.9: score += 10

        # 巴菲特指数（全市场市值/GDP，估值温度计）
        try:
            buffett = ak.stock_buffett_index_lg()
            if buffett is not None and len(buffett) > 0:
                b_val = float(buffett["value"].iloc[-1])
                signals.append(f"巴菲特指数: {b_val:.0f}%")
                if b_val > 100: score -= 10
                elif b_val < 60: score += 10
        except:
            pass

        # 股指期货基差
        try:
            basis_data = DataCollectAgent.get_futures_basis()
            if basis_data:
                basis_parts = [f"{k}基差: {v:+.2f}%" for k, v in basis_data.items()]
                signals.append(" | ".join(basis_parts))
                avg_basis = sum(basis_data.values()) / len(basis_data)
                if avg_basis > 0:
                    score += 5
                    signals.append(f"平均基差{avg_basis:+.2f}%, 升水(contango)偏多")
                else:
                    score -= 5
                    signals.append(f"平均基差{avg_basis:+.2f}%, 贴水(backwardation)偏空")
        except:
            pass

        score = float(np.clip(score, 0, 100))
        data_text = "\n".join(signals) if signals else "暂无实时跨市场数据"

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)
