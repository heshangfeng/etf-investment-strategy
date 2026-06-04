"""
ETF 智能投资分析系统 - 市场情绪/跨市场/游资/解禁/形态 Agent
"""
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from core.models import AgentReport
from core.data import DataCollectAgent
from agents.base import BaseLLMAgent
from infra.logger import get_logger; logger = get_logger(__name__)


# ====================== 【LLM多智能体 - 零售情绪结构】 ======================
class RetailSentimentAgent(BaseLLMAgent):
    ROLE_NAME = "零售情绪智能体"
    AGENT_TEMPERATURE = 0.5
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是行为金融学量化分析师，专注A股散户情绪测量。

分析框架：
1. 量比（成交量/20日均量）——放量过大=情绪过热（反向指标），缩量=冷清
2. 价格位置（52周高低位置）——接近高点=亢奋（反向），接近低点=恐慌（反向）
3. 两融余额变化——融资余额快速上升=散户加杠杆，情绪过热信号
4. 市场换手率——全市场换手率异常高=情绪极端
5. 散户情绪是典型的反向指标——极度乐观时见顶，极度悲观时见底
6. 量价背离——放量滞涨=出货信号，缩量下跌=杀跌动能衰竭
7. 散户情绪反向指标的权重不应超过整体判断的30%，需结合价格趋势验证

综合以上信号给出基于行为金融学的情绪判断。"""

    def _collect_market_data(self, etf_code: str) -> dict:
        df = DataCollectAgent.get_etf_price(etf_code)
        vol_20_mean = df["volume"].tail(20).mean()
        vol_today = df["volume"].iloc[-1]
        turnover_ratio = vol_today / vol_20_mean if vol_20_mean > 0 else 1.0
        close = df["close"].iloc[-1]
        high_52w = df["close"].tail(250).max() if len(df) >= 250 else df["close"].max()
        low_52w = df["close"].tail(250).min() if len(df) >= 250 else df["close"].min()
        pos_from_low = (close - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50

        margin_info = ""
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_m = margin['szse_margin'] + margin['sse_margin']
            margin_info = f"两融余额: {total_m:.0f}亿"
        except Exception as e:
            logger.warning("获取两融数据失败: " + str(e), exc_info=True)

        activity_info = ""
        try:
            act = ak.stock_market_activity_legu()
            if act is not None and "换手率" in act.columns:
                turnover_rate = float(act["换手率"].iloc[-1])
                activity_info = f"\n全市场换手率: {turnover_rate}%"
        except Exception as e:
            logger.warning("获取全市场活跃度失败: " + str(e), exc_info=True)

        bollinger_pct_b = None
        try:
            if len(df) >= 20:
                ma20_boll = df["close"].rolling(20).mean()
                std20 = df["close"].rolling(20).std()
                upper = ma20_boll + 2 * std20
                lower = ma20_boll - 2 * std20
                pct_b_val = (close - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
                if not np.isnan(pct_b_val):
                    bollinger_pct_b = float(pct_b_val)
        except Exception as e:
            logger.warning("计算布林带失败: " + str(e), exc_info=True)

        return {
            "df": df, "turnover_ratio": turnover_ratio, "pos_from_low": pos_from_low,
            "close": close, "margin_info": margin_info, "activity_info": activity_info,
            "bollinger_pct_b": bollinger_pct_b,
        }

    def _compute_retail_score(self, data: dict) -> tuple:
        score = 50.0
        signals = []
        tr = data["turnover_ratio"]
        pfl = data["pos_from_low"]
        boll = data["bollinger_pct_b"]

        if tr > 2.0:
            signals.append(f"放量异常(量比{tr:.2f})"); score -= 6
        elif tr > 1.5:
            signals.append(f"放量(量比{tr:.2f})"); score += 5
        elif tr < 0.5:
            signals.append(f"缩量(量比{tr:.2f})"); score -= 3
        else:
            signals.append(f"量能正常(量比{tr:.2f})")

        if pfl > 90:
            signals.append("接近年内高点"); score -= 5
        elif pfl < 10:
            signals.append("接近年内低点"); score += 5
        elif 40 <= pfl <= 60:
            signals.append("价格中位区间"); score += 3

        if boll is not None:
            if boll > 1.0:
                signals.append(f"布林带超买(%B={boll:.2f})"); score -= 4
            elif boll > 0.8:
                signals.append(f"布林带上轨附近(%B={boll:.2f})"); score -= 2
            elif boll < 0:
                signals.append(f"布林带超卖(%B={boll:.2f})"); score += 4
            elif boll < 0.2:
                signals.append(f"布林带下轨附近(%B={boll:.2f})"); score += 2

        return float(np.clip(score, 0, 100)), signals

    def _build_llm_prompt(self, etf_code: str, data: dict, score: float, signals: list) -> str:
        boll_str = f"\n布林带%B: {data['bollinger_pct_b']:.2f}" if data['bollinger_pct_b'] is not None else ""
        data_text = (f"量比(20日均值): {data['turnover_ratio']:.2f}\n"
                     f"52周价格位置: {data['pos_from_low']:.1f}%\n"
                     f"{boll_str}\n"
                     f"情绪信号: {'; '.join(signals)}\n"
                     f"{data['margin_info']}{data['activity_info']}")
        return self._enrich_with_memory(etf_code, data_text)

    def run(self, etf_code: str) -> AgentReport:
        data = self._collect_market_data(etf_code)
        score, signals = self._compute_retail_score(data)
        enriched_text = self._build_llm_prompt(etf_code, data, score, signals)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("市场情绪", etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        report = self._parse_to_report(etf_code, "市场情绪", llm_out, score, rating)
        report.data_summary = {"turnover_ratio": data["turnover_ratio"], "pos_from_low": data["pos_from_low"]}
        return report


# ====================== 【LLM多智能体 - 跨资产/跨市场联动】 ======================
class CrossMarketAgent(BaseLLMAgent):
    ROLE_NAME = "跨市场联动智能体"
    AGENT_TEMPERATURE = 0.6
    AGENT_USE_QUICK_MODEL = True

    _spot_cache = {}  # 类级别缓存，spot_quote API不稳定时使用

    SYSTEM_PROMPT = """你是全球宏观策略分析师，专注跨市场信号传导。

分析框架：
1. 人民币汇率(USDCNY)——升值利好A股（外资流入），贬值承压（外资流出）（重点关注）
2. 美债收益率(10Y)——全球资产定价锚，快速上行=风险资产承压（重点关注）
3. 美元指数——美元强弱影响全球资金流向新兴市场
4. 黄金价格——避险情绪指标，金价大涨=避险模式
5. 中美利差——利差倒挂时外资流出压力大
6. IVIX隐含波动率——A股恐慌指数，快速上升=市场恐慌
7. 巴菲特指数——全市场估值温度计

信号优先级：USDCNY > 美债收益率 > 中美利差 > IVIX > 黄金 > 美元指数
传导逻辑：美债利率+美元同时走强=新兴市场承压；人民币升值+美元弱=利好A股。
注意不同信号之间的传导有时滞，警惕冲突信号的市场含义。"""

    def _get_usdcny_signal(self) -> tuple:
        usdcny = None
        signal_text = ""
        score_adj = 0
        try:
            fx = ak.spot_quote()
            cny_row = fx[fx["名称"].str.contains("美元", na=False)]
            if len(cny_row) > 0:
                usdcny = float(cny_row["现价"].iloc[0])
                CrossMarketAgent._spot_cache["usdcny"] = usdcny
                signal_text = f"USDCNY: {usdcny}"
        except Exception:
            if "usdcny" in CrossMarketAgent._spot_cache:
                usdcny = CrossMarketAgent._spot_cache["usdcny"]
                signal_text = f"USDCNY: {usdcny}(缓存)"
            else:
                signal_text = "USDCNY: 暂无数据"

        if usdcny is not None:
            if usdcny > 7.3:
                score_adj -= 15
            elif usdcny < 6.9:
                score_adj += 10

        try:
            bond = DataCollectAgent.get_bond_yield()
            signal_text += f" | 中国10Y国债: {bond['cn_10y']}% | 美国10Y国债: {bond['us_10y']}% | 中美利差: {bond['spread']}%"
        except Exception as e:
            logger.warning("获取中美利差失败: " + str(e), exc_info=True)
        return signal_text, score_adj

    def _get_ivix_signal(self, etf_code: str) -> tuple:
        score_adj = 0
        try:
            ivix = DataCollectAgent.get_ivix(etf_code)
            if ivix is not None:
                signal_text = f"IVIX隐含波动率: {ivix:.1f}"
                if ivix > 30:
                    score_adj -= 8
                    signal_text += " (IVIX高位=市场恐慌)"
                elif ivix > 25:
                    score_adj -= 3
                elif ivix < 18:
                    score_adj += 3
                    signal_text += " (IVIX低位=市场平稳)"
                return signal_text, score_adj
        except Exception as e:
            logger.warning("获取IVIX隐含波动率失败: " + str(e), exc_info=True)
        return "", 0

    def _get_gold_signal(self) -> tuple:
        try:
            gold_df = ak.futures_zh_realtime()
            sym_col = "symbol" if "symbol" in gold_df.columns else "代码"
            price_col = "current_price" if "current_price" in gold_df.columns else "最新价"
            au_rows = gold_df[gold_df[sym_col].astype(str).str.contains("AU", na=False)]
            if len(au_rows) > 0:
                gold_price = float(au_rows[price_col].iloc[0])
                CrossMarketAgent._spot_cache["gold"] = gold_price
                text = f"黄金(AU): {gold_price:.0f}"
                score_adj = -5 if gold_price > 600 else 0
                if score_adj:
                    text += " (金价高位=避险升温)"
                return text, score_adj
        except Exception:
            pass
        if "gold" in CrossMarketAgent._spot_cache:
            gold_price = CrossMarketAgent._spot_cache["gold"]
            return f"黄金(AU): {gold_price:.0f}(缓存)", 0
        return "", 0

    def _get_buffett_signal(self) -> tuple:
        try:
            buffett = ak.stock_buffett_index_lg()
            if buffett is not None and len(buffett) > 0:
                b_val = float(buffett["value"].iloc[-1])
                text = f"巴菲特指数: {b_val:.0f}%"
                score_adj = 0
                if b_val > 100:
                    score_adj -= 10
                elif b_val < 60:
                    score_adj += 10
                return text, score_adj
        except Exception as e:
            logger.warning("获取巴菲特指数失败: " + str(e), exc_info=True)
        return "", 0

    def _get_futures_basis_signal(self) -> tuple:
        try:
            basis_data = DataCollectAgent.get_futures_basis()
            if basis_data:
                basis_parts = [f"{k}基差: {v:+.2f}%" for k, v in basis_data.items()]
                text = " | ".join(basis_parts)
                avg_basis = sum(basis_data.values()) / len(basis_data)
                if avg_basis > 0:
                    score_adj = 5
                    text += f"\n平均基差{avg_basis:+.2f}%, 升水(contango)偏多"
                else:
                    score_adj = -5
                    text += f"\n平均基差{avg_basis:+.2f}%, 贴水(backwardation)偏空"
                return text, score_adj
        except Exception as e:
            logger.warning("获取股指期货基差失败: " + str(e), exc_info=True)
        return "", 0

    def run(self, etf_name: str, etf_code: str) -> AgentReport:
        score = 50.0
        signals = []

        txt, adj = self._get_usdcny_signal()
        signals.append(txt)
        score += adj

        txt, adj = self._get_ivix_signal(etf_code)
        if txt:
            signals.append(txt)
            score += adj

        txt, adj = self._get_gold_signal()
        if txt:
            signals.append(txt)
            score += adj

        txt, adj = self._get_buffett_signal()
        if txt:
            signals.append(txt)
            score += adj

        txt, adj = self._get_futures_basis_signal()
        if txt:
            signals.append(txt)
            score += adj

        score = float(np.clip(score, 0, 100))
        data_text = "\n".join(signals) if signals else "暂无实时跨市场数据"
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 游资情绪热点】 ======================
class HotMoneyAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "游资情绪智能体"
    AGENT_USE_QUICK_MODEL = True
    AGENT_TEMPERATURE = 0.4
    SYSTEM_PROMPT = """你是专注A股短线情绪的游资分析师，擅长识别市场极端情绪。
分析框架：
1. 涨停数量反映短线情绪温度——涨停>80=情绪过热（反向指标），涨停<20=情绪冰点（反弹机会）
2. 连板高度（最高连板数）——高度板>7=情绪极端亢奋，容易见顶回落
3. 涨停封板资金——封板资金大=做多意愿强，封板资金小=容易炸板
4. 游资情绪是典型的反向指标——极度亢奋时见顶，极度恐慌时见底
5. 涨停行业集中度——集中在单一板块=结构性行情，分散=普涨格局
给出基于游资情绪的独立判断。"""

    _zt_cache = None  # 类级别缓存，避免重复请求
    _zt_cache_time = None  # 缓存时间戳
    _zt_last_success = None  # 最近一次成功获取的数据（API失败时使用）

    @classmethod
    def _get_zt_data(cls) -> dict:
        """获取涨停数据（带类级别缓存 + 过期刷新 + 失败保留旧数据）。"""
        now = datetime.now()
        # 如果有缓存且未超过1小时，直接返回
        if cls._zt_cache is not None and cls._zt_cache_time is not None:
            elapsed = (now - cls._zt_cache_time).total_seconds()
            if elapsed < 3600:
                return cls._zt_cache
        result = {"limit_up": -1, "max_consecutive": 0, "industries": ""}
        try:
            zt_df = ak.stock_zt_pool_em()
            if zt_df is not None and len(zt_df) > 0:
                result["limit_up"] = len(zt_df)
                # 连板高度
                if "连板数" in zt_df.columns:
                    board = zt_df["连板数"].dropna().astype(int)
                    result["max_consecutive"] = int(board.max()) if len(board) > 0 else 0
                # 行业分布
                if "行业" in zt_df.columns:
                    ind_count = zt_df["行业"].value_counts().head(5)
                    result["industries"] = ", ".join(f"{k}({v}家)" for k, v in ind_count.items())
                cls._zt_last_success = dict(result)
        except Exception:
            # API失败时，使用上次成功的数据（即便已过期）
            if cls._zt_last_success is not None:
                result = cls._zt_last_success
                result["from_cache"] = True
                cls._zt_cache = result
                cls._zt_cache_time = now
                return result
        cls._zt_cache = result
        cls._zt_cache_time = now
        return result

    def run(self, etf_code: str) -> AgentReport:
        zt = self._get_zt_data()
        limit_up = zt["limit_up"]
        max_board = zt["max_consecutive"]
        industries = zt["industries"]

        if limit_up < 0:
            score = 50.0
            data_text = "暂无实时涨停数据"
        else:
            # 反向指标：涨停过多=过热看空，涨停过少=冰点看多
            score = 50.0
            sentiment_desc = ""
            if limit_up > 100:
                score = 20
                sentiment_desc = "情绪极度亢奋（反向看空）"
            elif limit_up > 80:
                score = 30
                sentiment_desc = "情绪过热（反向偏空）"
            elif limit_up > 50:
                score = 45
                sentiment_desc = "情绪偏暖"
            elif limit_up > 30:
                score = 55
                sentiment_desc = "情绪正常"
            elif limit_up > 15:
                score = 65
                sentiment_desc = "情绪偏冷（反向偏多）"
            else:
                score = 75
                sentiment_desc = "情绪冰点（反向看多）"

            # 连板高度惩罚
            if max_board >= 7:
                score -= 10
                sentiment_desc += f"，连板{max_board}板风险高"
            elif max_board >= 5:
                score -= 5

            score = float(np.clip(score, 0, 100))

            data_text = (f"今日涨停家数: {limit_up}\n"
                         f"最高连板: {max_board}板\n"
                         f"涨停行业分布: {industries or '未知'}\n"
                         f"情绪判断: {sentiment_desc}")

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("游资情绪", etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, "游资情绪", llm_out, score, rating)


# ====================== 【LLM多智能体 - 解禁压力预警】 ======================
class UnlockPressureAgent(BaseLLMAgent):
    ROLE_NAME = "解禁压力智能体"
    AGENT_USE_QUICK_MODEL = True
    AGENT_TEMPERATURE = 0.3
    SYSTEM_PROMPT = """你是专注解禁压力分析的策略分析师。
分析框架：
1. 未来30天解禁总市值——>500亿=显著压力，>1000亿=严重压力
2. 解禁市值占流通市值比例——比例越高，抛压越大
3. 解禁日期临近程度——距离越近，市场提前反应的概率越大
4. 首发原股东限售股解禁影响最大，定向增发次之
5. 大规模解禁前后1-2周市场往往承压，但有时提前消化后反而是机会
6. 当解禁数据不可获取时，默认保持中性立场（50分），不以暂无数据作为看多依据
给出基于解禁压力的风险评估，高分=解禁压力小（安全），低分=解禁压力大（风险）。"""

    _unlock_cache = None  # 类级别缓存
    _last_result = None   # 最近一次成功获取的结果

    @classmethod
    def _get_unlock_data(cls) -> dict:
        """获取解禁数据（带缓存）。失败时保留上次有效结果。"""
        if cls._unlock_cache is not None:
            return cls._unlock_cache
        result = {"total_value": 0, "count": 0, "upcoming": "", "data_ok": False}
        try:
            restricted = ak.stock_restricted_release_queue_sina()
            if restricted is not None and len(restricted) > 0:
                # 未来30天解禁
                now = datetime.now()
                cutoff = now + timedelta(days=30)
                total_val = 0.0
                count = 0
                upcoming_dates = {}
                for _, row in restricted.iterrows():
                    try:
                        dt_str = str(row.get("解禁日期", ""))
                        dt = datetime.strptime(dt_str, "%Y-%m-%d") if dt_str else None
                        if dt and now <= dt <= cutoff:
                            val = float(row.get("解禁市值(元)", 0))
                            total_val += val
                            count += 1
                            dkey = dt_str[:10]
                            upcoming_dates[dkey] = upcoming_dates.get(dkey, 0) + 1
                    except Exception as e:
                        logger.warning("解析解禁数据行失败: " + str(e), exc_info=True)
                        pass
                result["total_value"] = total_val
                result["count"] = count
                result["data_ok"] = True
                date_summary = "; ".join(f"{d}({n}只)" for d, n in sorted(upcoming_dates.items())[:5])
                result["upcoming"] = date_summary
        except Exception as e:
            logger.warning("获取解禁数据失败: " + str(e), exc_info=True)
            pass
        cls._unlock_cache = result
        if result["data_ok"]:
            cls._last_result = dict(result)
        return result

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        unlock = self._get_unlock_data()
        total_val = unlock["total_value"]
        unlock_count = unlock["count"]
        upcoming = unlock["upcoming"]

        score = 50.0
        if unlock.get("data_ok") and total_val > 0:
            total_val_yi = total_val / 1e8  # 转亿元
            if total_val_yi > 1000:
                score = 20
                pressure = "严重解禁压力"
            elif total_val_yi > 500:
                score = 35
                pressure = "显著解禁压力"
            elif total_val_yi > 200:
                score = 50
                pressure = "中等解禁压力"
            elif total_val_yi > 50:
                score = 65
                pressure = "轻度解禁压力"
            else:
                score = 80
                pressure = "解禁压力较小"
            data_text = (f"未来30天解禁总市值: {total_val_yi:.0f}亿\n"
                         f"解禁股票数量: {unlock_count}只\n"
                         f"解禁日期分布: {upcoming or '未知'}\n"
                         f"压力评估: {pressure}")
        elif unlock.get("data_ok"):
            data_text = "暂无近期解禁数据或解禁压力较小"
            score = 65
        else:
            data_text = "数据获取失败，中性评估"
            score = 50

        score = float(np.clip(score, 0, 100))
        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 技术形态识别】 ======================
class PatternRecognitionAgent(BaseLLMAgent):
    ROLE_NAME = "技术形态智能体"
    AGENT_USE_QUICK_MODEL = True
    AGENT_TEMPERATURE = 0.4
    SYSTEM_PROMPT = """你是拥有20年经验的技术形态识别专家，擅长从K线图中识别经典形态。

重要声明：技术形态识别本质上是概率性的，单一形态不应作为独立交易依据。形态信号需要成交量确认和趋势强度验证。

分析框架：
1. 经典反转形态：头肩顶/底、双顶/底、圆弧顶/底、V型反转
2. 经典持续形态：旗形、三角旗形、楔形、矩形
3. 关键支撑/阻力位：前期高低点、密集成交区、整数关口
4. 趋势强度：通过ADX衡量趋势强度（<20=弱趋势, 20-40=中等趋势, >40=强趋势）
5. K线组合：启明星/黄昏星、吞没形态、十字星、锤子线/上吊线
6. 突破确认：形态突破需要成交量配合，假突破是常见陷阱
7. 趋势确认：ADX+成交量双重确认信号才有高可信度

先识别当前最可能的技术形态，再给出基于形态的目标位和止损位。
每次输出需包含对信号可信度的评估（低/中/高），低于中等可信度时应标注为参考信号。"""

    def _detect_trend_lines(self, code: str, price_data: dict) -> tuple:
        close = price_data["close"]
        n = price_data["n"]
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        try:
            slope, intercept = np.linalg.lstsq(A, close, rcond=None)[0]
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            r2 = ss_res / max(ss_tot, 1e-10)
            trend_strength = 1 - r2
            if slope > 0:
                pattern = f"上升趋势(斜率{slope:.4f}, R²={1-r2:.2f})"
                score_delta = 10 * trend_strength
            else:
                pattern = f"下降趋势(斜率{slope:.4f}, R²={1-r2:.2f})"
                score_delta = -10 * trend_strength
            return score_delta, [pattern]
        except Exception as e:
            logger.warning("线性回归计算趋势失败: " + str(e), exc_info=True)
        return 0, []

    def _compute_adx_strength(self, code: str, price_data: dict) -> tuple:
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        n = price_data["n"]
        try:
            if n >= 16:
                tr = np.zeros(n)
                for i in range(1, n):
                    hl = high[i] - low[i]
                    hc = abs(high[i] - close[i - 1])
                    lc = abs(low[i] - close[i - 1])
                    tr[i] = max(hl, hc, lc)
                plus_dm = np.zeros(n)
                minus_dm = np.zeros(n)
                for i in range(1, n):
                    up_move = high[i] - high[i - 1]
                    down_move = low[i - 1] - low[i]
                    if up_move > down_move and up_move > 0:
                        plus_dm[i] = up_move
                    if down_move > up_move and down_move > 0:
                        minus_dm[i] = down_move
                period = 14
                atr = np.mean(tr[1:period + 1])
                plus_smooth = np.mean(plus_dm[1:period + 1])
                minus_smooth = np.mean(minus_dm[1:period + 1])
                for i in range(period + 1, n):
                    atr = (atr * (period - 1) + tr[i]) / period
                    plus_smooth = (plus_smooth * (period - 1) + plus_dm[i]) / period
                    minus_smooth = (minus_smooth * (period - 1) + minus_dm[i]) / period
                if atr > 0:
                    plus_di = 100 * plus_smooth / atr
                    minus_di = 100 * minus_smooth / atr
                    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
                    desc = "强趋势" if dx > 40 else "中等趋势" if dx > 20 else "弱趋势"
                    return 0, [f"ADX趋势强度={dx:.1f}({desc})"]
        except Exception as e:
            logger.warning("ADX趋势强度计算失败: " + str(e), exc_info=True)
        return 0, []

    def _detect_support_resistance(self, code: str, price_data: dict) -> tuple:
        low = price_data["low"]
        high = price_data["high"]
        patterns = []
        support_levels = []
        resistance_levels = []
        for level_pct in [10, 20, 30, 40]:
            threshold = np.percentile(low, level_pct)
            touches = np.sum(np.abs(low - threshold) / threshold < 0.02)
            if touches >= 3:
                support_levels.append((threshold, int(touches)))
        for level_pct in [60, 70, 80, 90]:
            threshold = np.percentile(high, level_pct)
            touches = np.sum(np.abs(high - threshold) / threshold < 0.02)
            if touches >= 3:
                resistance_levels.append((threshold, int(touches)))
        if support_levels:
            best_support = min(support_levels, key=lambda x: x[0])
            patterns.append(f"支撑位: {best_support[0]:.3f}(测试{best_support[1]}次)")
        if resistance_levels:
            best_res = max(resistance_levels, key=lambda x: x[0])
            patterns.append(f"阻力位: {best_res[0]:.3f}(测试{best_res[1]}次)")
        return 0, patterns

    def _detect_double_top_bottom(self, code: str, price_data: dict) -> tuple:
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        vol = price_data["volume"]
        n = price_data["n"]
        mid = n // 2
        score_delta = 0
        patterns = []
        left_max = np.max(high[:mid])
        right_max = np.max(high[mid:])
        left_min = np.min(low[:mid])
        right_min = np.min(low[mid:])
        vol_confirm_high = vol_confirm_low = ""
        if vol is not None:
            vol_left_mean = np.mean(vol[:mid])
            vol_right_mean = np.mean(vol[mid:])
        else:
            vol_left_mean = vol_right_mean = 0
        if abs(left_max - right_max) / max(left_max, right_max) < 0.03 and left_max > np.median(close):
            if vol is not None and vol_right_mean < vol_left_mean * 0.9:
                vol_confirm_high = " (右顶缩量确认)"
            patterns.append(f"疑似双顶形态(L:{left_max:.3f}, R:{right_max:.3f}){vol_confirm_high}")
            score_delta -= 5
        if abs(left_min - right_min) / max(left_min, right_min) < 0.03 and left_min < np.median(close):
            if vol is not None and vol_right_mean > vol_left_mean * 1.1:
                vol_confirm_low = " (右底放量确认)"
            patterns.append(f"疑似双底形态(L:{left_min:.3f}, R:{right_min:.3f}){vol_confirm_low}")
            score_delta += 5
        return score_delta, patterns

    def _detect_candlestick_patterns(self, code: str, price_data: dict) -> tuple:
        close = price_data["close"]
        n = price_data["n"]
        score_delta = 0
        patterns = []
        if n >= 3:
            c1, c2, c3 = close[-3], close[-2], close[-1]
            o1, o2, o3 = close[-4] if n >= 4 else c1, c1, c2
            if c1 < o1 * 0.97 and abs(c2 - o2) / o2 < 0.01 and c3 > o3 * 1.03:
                patterns.append("启明星形态(看涨)")
                score_delta += 5
            elif c1 > o1 * 1.03 and abs(c2 - o2) / o2 < 0.01 and c3 < o3 * 0.97:
                patterns.append("黄昏星形态(看跌)")
                score_delta -= 5
            ret_3 = (close[-1] / close[-3] - 1) * 100
            ret_2 = (close[-1] / close[-2] - 1) * 100
            if ret_3 > 3 and ret_2 > 0:
                patterns.append("近3日连续上涨")
                score_delta += 5
            elif ret_3 < -3 and ret_2 < 0:
                patterns.append("近3日连续下跌")
                score_delta -= 5
        return score_delta, patterns

    def run(self, etf_code: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 30:
            return AgentReport(
                self.ROLE_NAME, etf_code, "技术形态",
                "中性", 50,
                "数据不足30个交易日，无法进行形态识别",
                ["数据不足"], [], 0.3, {}, "rule_fallback"
            )

        close = df["close"].values
        high = df["high"].values if "high" in df.columns else df["close"].values
        low = df["low"].values if "low" in df.columns else df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.zeros(len(df))

        n = min(60, len(close))
        price_data = {
            "close": close[-n:], "high": high[-n:], "low": low[-n:],
            "volume": volume[-n:], "n": n,
        }

        score = 50.0
        patterns_found = []

        delta, pats = self._detect_trend_lines(etf_code, price_data)
        score += delta
        patterns_found.extend(pats)

        delta, pats = self._compute_adx_strength(etf_code, price_data)
        score += delta
        patterns_found.extend(pats)

        delta, pats = self._detect_support_resistance(etf_code, price_data)
        score += delta
        patterns_found.extend(pats)

        delta, pats = self._detect_double_top_bottom(etf_code, price_data)
        score += delta
        patterns_found.extend(pats)

        delta, pats = self._detect_candlestick_patterns(etf_code, price_data)
        score += delta
        patterns_found.extend(pats)

        score = float(np.clip(score, 0, 100))

        close_60 = price_data["close"]
        chart_lines = []
        for i in range(n):
            bar = "↑" if close_60[i] > close_60[i - 1] else "↓" if i > 0 else "─"
            chart_lines.append(f"  [{i+1:2d}] 收{close_60[i]:.4f} {bar}")
        chart_str = "\n".join(chart_lines[-20:])

        data_text = (f"【最近{n}个交易日价格序列】\n{chart_str}\n\n"
                     f"【技术形态识别】\n")
        if patterns_found:
            for p in patterns_found:
                data_text += f"  • {p}\n"
        else:
            data_text += "  无明显经典形态\n"

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("技术形态", etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, "技术形态", llm_out, score, rating)


# ====================== 【LLM多智能体 - 趋势预测】 ======================
class TrendPredictorAgent(BaseLLMAgent):
    """基于 LLM 的趋势预测智能体（替代 Kronos，轻量级）。专注短期价格动量。"""
    ROLE_NAME = "趋势预测智能体"
    AGENT_TEMPERATURE = 0.3
    AGENT_USE_QUICK_MODEL = True
    SYSTEM_PROMPT = """你是专注短期动量交易的量化趋势预测分析师。

本智能体专注短期(5-20日)价格趋势预测，不分析行业基本面。

分析框架（权重由高到低）：
1. 短期动量（5日）：基于最近价格涨跌幅、加速度变化——核心信号
2. 均线系统：MA5/MA20排列关系，金叉/死叉的短期指引
3. 成交量验证：放量上涨/下跌的持续性评估，缩量反转信号
4. 波动率分析：高波动=趋势不稳，低波动=趋势延续
5. 动量衰减/背离：价格创新高但涨幅收窄=动能衰竭（反转信号）
6. 概率评估：给出未来5日和20日的方向概率及置信度

与行业分析不同，本智能体完全基于价格和成交量的统计规律，不做基本面归因。
请仔细分析价格数据，给出基于短期动量规律的预测。"""

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        n = len(df)
        if n < 30:
            return AgentReport(self.ROLE_NAME, etf_code, etf_name, "中性", 50,
                               "数据不足30个交易日", ["数据不足"], [], 0.3, {}, "rule_fallback")

        close = df["close"].values
        volume = df["volume"].values

        # ── 规则评分 ──
        score = 50.0
        signals = []

        # 1. 短期动量（近5日）
        ret_5d = close[-1] / close[-5] - 1 if n >= 5 else 0
        if ret_5d > 0.02:
            score += 8
            signals.append(f"近5日上涨{ret_5d*100:.1f}%")
        elif ret_5d < -0.02:
            score -= 8
            signals.append(f"近5日下跌{ret_5d*100:.1f}%")

        # 2. 中期动量（近20日）
        ret_20d = close[-1] / close[-20] - 1 if n >= 20 else 0
        if ret_20d > 0.05:
            score += 5
        elif ret_20d < -0.05:
            score -= 5

        # 3. 均线位置
        ma5 = np.mean(close[-5:]) if n >= 5 else close[-1]
        ma20 = np.mean(close[-20:]) if n >= 20 else close[-1]
        if close[-1] > ma5 and ma5 > ma20:
            score += 10
            signals.append("多头排列")
        elif close[-1] < ma5 and ma5 < ma20:
            score -= 10
            signals.append("空头排列")
        else:
            signals.append("均线交织")

        # 4. 成交量验证（近5日量 vs 近20日量）
        vol_5 = np.mean(volume[-5:]) if n >= 5 else 0
        vol_20 = np.mean(volume[-20:]) if n >= 20 else vol_5
        vol_ratio = vol_5 / max(vol_20, 1)
        if vol_ratio > 1.5 and ret_5d > 0:
            score += 5
            signals.append("放量上涨")
        elif vol_ratio > 1.5 and ret_5d < 0:
            score -= 5
            signals.append("放量下跌")

        # 5. 波动率
        vola = np.std(close[-20:] / close[-21:-1]) * 100 if n >= 21 else 0
        if vola > 3:
            score -= 5
            signals.append(f"高波动({vola:.1f}%)")
        elif vola < 1:
            score += 3
            signals.append(f"低波动({vola:.1f}%)")

        score = float(np.clip(score, 0, 100))

        # 构建未来5日、20日的简单预测
        pred_5d = f"预测未来5日: {'上涨' if ret_5d > 0 else '下跌'}趋势延续概率"
        pred_20d = f"预测未来20日: {'偏多' if ret_20d > 0 else '偏空'}格局"
        data_text = (f"近5日涨跌: {ret_5d*100:.2f}%\n"
                     f"近20日涨跌: {ret_20d*100:.2f}%\n"
                     f"均线状态: {'; '.join(signals)}\n"
                     f"波动率: {vola:.2f}%\n"
                     f"{pred_5d}\n{pred_20d}")

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)
