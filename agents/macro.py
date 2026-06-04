"""
ETF 智能投资分析系统 - 宏观/货币/政策 Agent
"""
import akshare as ak
import numpy as np
from datetime import datetime

from core.config import LLM_ENABLED
from core.models import AgentReport
from core.data import DataCollectAgent
from agents.base import BaseLLMAgent


# ====================== 【LLM多智能体 - 宏观分析】 ======================
class MacroAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "宏观分析智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是拥有20年经验的央行宏观分析师，你的分析覆盖经济全貌而不只看成交量。

分析框架（综合以下维度）：
1. 市场成交量代表流动性和参与度——成交额>10000亿=强趋势市场，7000-10000亿=震荡，<7000亿=弱势
2. 全市场估值温度——沪深300 PE百分位所处区间决定系统性机会/风险
3. 宏观经济景气度——PMI趋势反映经济基本面强弱
4. 信用周期——社融/M2增速变化预示流动性方向
5. 通胀环境——CPI/PPI走势影响货币政策空间
6. 全球风险偏好——VIX指数、北向资金流向反映外资态度

务必综合多维度判断，不要只看成交额。简洁专业，数据驱动。"""

    def run(self) -> AgentReport:
        vol = DataCollectAgent.get_market_total_volume()

        # 收集更多宏观数据
        extra = []
        try:
            val = DataCollectAgent.get_index_val("000300")
            extra.append(f"沪深300 PE百分位: {val['pe_percent']}%")
        except:
            pass
        try:
            pmi_df = ak.macro_china_pmi()
            if pmi_df is not None and len(pmi_df) > 0:
                pmi_val = float(pmi_df.tail(1).values[0][1])
                extra.append(f"制造业PMI: {pmi_val}")
        except:
            pass
        try:
            vix_val = DataCollectAgent.get_ivix("510300")
            extra.append(f"VIX: {vix_val}")
        except:
            pass

        extra_text = " | ".join(extra) if extra else ""
        data_text = f"今日全市场成交额：{vol:.0f}亿元 | {extra_text}"

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
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
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

        # 获取最新政策新闻（多源）
        policy_news = ""
        try:
            from core.data import PublicOpinionAgent
            kw_list = ["宏观经济 政策", "政治局会议", "国务院 政策", "金融监管",
                       "货币政策", "房地产 政策", "资本市场 改革"]
            news_parts = []
            for kw in kw_list:
                n = PublicOpinionAgent.get_professional_news(kw)
                if n and "暂无" not in n and "公开财经资讯" not in n:
                    news_parts.append(f"[{kw}] {n[:150]}")
            # 补充akshare财新头条（不依赖关键词匹配）
            try:
                import akshare as ak
                cx = ak.stock_news_main_cx()
                if cx is not None and len(cx) > 0:
                    for _, row in cx.head(3).iterrows():
                        news_parts.append(f"[财新] {row['summary'][:100]}")
            except Exception:
                pass
            if news_parts:
                policy_news = "【近期政策资讯】\n" + "\n".join(news_parts[:5]) + "\n\n"
        except Exception:
            pass

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

        data_text = (f"{policy_news}"
                     f"当前日期: {now.strftime('%Y-%m-%d')}\n"
                     f"季节效应: {season_effect}\n"
                     f"会议窗口: {current_event or '无重要会议窗口'}\n"
                     f"月份特征: {month}月")

        score = 50.0
        if current_event: score += 10
        if month in (3, 7, 12): score += 8
        if month in (1, 2): score += 5
        score = float(np.clip(score, 0, 100))

        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国政策周期", "POLICY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        report = self._parse_to_report("POLICY", "政策周期", llm_out, score, rating)
        report.data_summary = {"current_event": current_event, "season_effect": season_effect, "month": month}
        return report
