"""
ETF 智能投资分析系统 - 基本面 Agent（价值/技术/情绪/资金/风控/行业）
"""
import numpy as np
import pandas as pd
import akshare as ak
from core.config import CACHE_OPINION
from core.models import AgentReport
from core.data import DataCollectAgent, PublicOpinionAgent
from agents.base import BaseLLMAgent
from core.keywords import INDUSTRY_POS


# ====================== 【LLM多智能体 - 价值估值】 ======================
class ValueAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "价值估值智能体"
    AGENT_TEMPERATURE = 0.3
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是信奉格雷厄姆-巴菲特价值投资理念的分析师。

分析框架：
1. PE/PB百分位是核心——百分位<30%为低估区域，>70%为高估区域
2. 百分位在30%-70%之间为估值合理区域，需结合其他维度判断
3. 估值百分位结合绝对估值水平——PE绝对值过高时即使百分位中等也需谨慎
4. 行业估值分化——不同行业的合理PE/PB范围不同，不能跨行业简单比较
5. 业绩增长消化估值——高增长行业可接受更高估值
6. 安全边际原则——在低估区间买入提供下行保护，在高估区间卖出锁定收益

给出基于估值安全边际的独立判断。"""

    def run(self, etf_code: str, etf_name: str, index_code: str) -> AgentReport:
        d = DataCollectAgent.get_index_val(index_code)
        score = round(100 - d["pe_percent"], 2)
        data_text = f"PE(TTM): {d['pe']} | PB: {d['pb']} | PE近5年百分位: {d['pe_percent']}%"
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 技术趋势】 ======================
class TechAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "技术趋势智能体"
    AGENT_TEMPERATURE = 0.3
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是拥有15年经验的技术分析师，擅长综合多种技术指标判断趋势。

分析框架（综合多个时间维度）：
1. 均线系统：价格与MA5/MA20的关系判断短期趋势方向
2. 均线交叉：MA5与MA20金叉=看多信号，死叉=看空信号
3. RSI(6日)：>70=超买（可能回调），<30=超卖（可能反弹）
4. MACD：DIF与DEA金叉=动能转多，死叉=动能转空，柱状体变化=动能强弱
5. 布林带：价格触及上轨=超买，触及下轨=超卖，带宽收窄=变盘信号
6. 成交量确认：上涨需放量确认，下跌放量=恐慌，缩量下跌=跌势将尽

Warning: RSI > 70 with bearish divergence is a strong sell signal
Warning: MACD柱状体缩头/缩脚是动能衰减的关键信号
Warning: 多种指标共振时信号更可靠，单一指标不可过度依赖

结合价格位置、均线形态、技术指标和成交量给出综合判断。注意不同指标信号矛盾时的处理。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 20:
            return AgentReport(self.ROLE_NAME, etf_code, etf_name, "中性", 50,
                               "数据不足20个交易日", ["数据不足"], [], 0.3, {}, "rule_fallback")
        close, m5, m20 = df["close"].iloc[-1], df["ma5"].iloc[-1], df["ma20"].iloc[-1]

        # === RSI(6) ===
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(6).mean()
        avg_loss = loss.rolling(6).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # === MACD (12, 26, 9) ===
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        dif_val, dea_val = dif.iloc[-1], dea.iloc[-1]
        macd_cross = "金叉" if dif_val > dea_val else "死叉" if dif_val < dea_val else "粘合"

        # === Bollinger Bands (20, 2) ===
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_width = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid * 100
        bb_w = bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.0

        # === 规则评分（均线 + RSI + MACD 共振） ===
        ts = 50
        if close > m5 and close > m20: ts += 18
        if m5 > m20: ts += 12
        if close < m5 and close < m20: ts -= 30
        if rsi_val < 30: ts += 10       # 超卖反弹
        elif rsi_val > 70: ts -= 10     # 超买回调
        if dif_val > dea_val: ts += 8   # MACD金叉
        elif dif_val < dea_val: ts -= 8 # MACD死叉
        score = float(np.clip(ts, 0, 100))

        recent_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        rsi_signal = "超买区" if rsi_val > 70 else "超卖区" if rsi_val < 30 else "中性"
        data_text = (f"最新价: {close:.3f} | MA5: {m5:.3f} | MA20: {m20:.3f}\n"
                     f"均线关系: {'多头' if m5 > m20 else '空头'}排列\n"
                     f"近5日涨跌: {recent_5d:.2f}%\n"
                     f"近5日平均波动率: {df['volatility'].tail(5).mean():.4f}\n"
                     f"RSI(6): {rsi_val:.1f} ({rsi_signal})\n"
                     f"MACD: DIF={dif_val:.4f} DEA={dea_val:.4f} 柱={ (dif - dea).iloc[-1] * 2 :.4f} | 信号: {macd_cross}\n"
                     f"布林带宽: {bb_w:.1f}%")
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 舆情情绪】 ======================
class SentimentAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "舆情情绪智能体"
    AGENT_TEMPERATURE = 0.5
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是行为金融学专家，擅长识别市场情绪。

分析框架：
1. 情绪分数>70为乐观（可能过度乐观，反向信号），<30为恐慌（可能过度悲观）
2. 舆情趋势比单日分数更重要——持续回暖或持续走弱是强烈信号
3. 结合行业关键词判断是否存在实质性利好/利空
4. 关注多空交织情况——矛盾信号意味着市场分歧大
5. 结合全市场情绪——当市场整体极度悲观时，个股情绪的参考价值下降
6. 情绪极端值往往对应行情拐点——极端恐慌是买入机会，极端乐观是卖出时机

重点关注情绪极端值和趋势变化，给出逆势判断。"""

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
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        score = op['opinion_score']
        if score >= 80: fr = "强烈看多"
        elif score >= 65: fr = "看多"
        elif score >= 45: fr = "中性"
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
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
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
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
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

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)
