"""
ETF 智能投资分析系统 - 市场情绪/跨市场/游资/解禁/形态 Agent
"""
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from models import AgentReport
from data import DataCollectAgent
from agents.base import BaseLLMAgent


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

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("市场情绪", etf_code, enriched_text))
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

        enriched_text = self._enrich_with_memory(etf_code, data_text)
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, enriched_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
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

    @classmethod
    def _get_zt_data(cls) -> dict:
        """获取涨停数据（带类级别缓存）。"""
        if cls._zt_cache is not None:
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
        except Exception:
            pass
        cls._zt_cache = result
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
给出基于解禁压力的风险评估，高分=解禁压力小（安全），低分=解禁压力大（风险）。"""

    _unlock_cache = None  # 类级别缓存

    @classmethod
    def _get_unlock_data(cls) -> dict:
        """获取解禁数据（带缓存）。"""
        if cls._unlock_cache is not None:
            return cls._unlock_cache
        result = {"total_value": 0, "count": 0, "upcoming": ""}
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
                    except:
                        pass
                result["total_value"] = total_val
                result["count"] = count
                date_summary = "; ".join(f"{d}({n}只)" for d, n in sorted(upcoming_dates.items())[:5])
                result["upcoming"] = date_summary
        except Exception:
            pass
        cls._unlock_cache = result
        return result

    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        unlock = self._get_unlock_data()
        total_val = unlock["total_value"]
        unlock_count = unlock["count"]
        upcoming = unlock["upcoming"]

        score = 50.0
        if total_val > 0:
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
        else:
            data_text = "暂无近期解禁数据或解禁压力较小"
            score = 65

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
    SYSTEM_PROMPT = """你是拥有20年经验的技术形态识别专家。
分析框架：
1. 识别经典K线形态：双顶/双底、头肩顶/底、旗形整理、三角整理
2. 支撑位和阻力位——价格反复测试的关键水平位
3. 趋势强度——线性回归斜率+RSquared判断趋势可信度
4. 近期K线组合——启明星/黄昏星、吞没形态、十字星等反转信号
结合价格位置和形态特征给出技术判断。
注意：高分=看涨形态（双底、突破阻力等），低分=看跌形态（双顶、跌破支撑等）。"""

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
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        # 取最近60根K线
        n = min(60, len(close))
        close_60 = close[-n:]
        high_60 = high[-n:]
        low_60 = low[-n:]

        score = 50.0
        patterns_found = []
        support_levels = []
        resistance_levels = []

        # ── 1. 趋势强度分析（线性回归） ──
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        try:
            slope, intercept = np.linalg.lstsq(A, close_60, rcond=None)[0]
            y_pred = slope * x + intercept
            ss_res = np.sum((close_60 - y_pred) ** 2)
            ss_tot = np.sum((close_60 - np.mean(close_60)) ** 2)
            r2 = ss_res / max(ss_tot, 1e-10)
            trend_strength = 1 - r2  # 1=完美趋势, 0=无趋势
            if slope > 0:
                patterns_found.append(f"上升趋势(斜率{slope:.4f}, R²={1-r2:.2f})")
                score += 10 * trend_strength
            else:
                patterns_found.append(f"下降趋势(斜率{slope:.4f}, R²={1-r2:.2f})")
                score -= 10 * trend_strength
        except Exception:
            pass

        # ── 2. 局部极值检测（寻找支撑/阻力） ──
        half = n // 2
        left_half = close_60[:half]
        right_half = close_60[half:]

        # 支撑：左半最低价附近反复测试
        for level_pct in [10, 20, 30, 40]:
            threshold = np.percentile(low_60, level_pct)
            touches = np.sum(np.abs(low_60 - threshold) / threshold < 0.02)
            if touches >= 3:
                support_levels.append((threshold, int(touches)))

        # 阻力：右半最高价附近反复测试
        for level_pct in [60, 70, 80, 90]:
            threshold = np.percentile(high_60, level_pct)
            touches = np.sum(np.abs(high_60 - threshold) / threshold < 0.02)
            if touches >= 3:
                resistance_levels.append((threshold, int(touches)))

        if support_levels:
            best_support = min(support_levels, key=lambda x: x[0])
            patterns_found.append(f"支撑位: {best_support[0]:.3f}(测试{best_support[1]}次)")

        if resistance_levels:
            best_res = max(resistance_levels, key=lambda x: x[0])
            patterns_found.append(f"阻力位: {best_res[0]:.3f}(测试{best_res[1]}次)")

        # ── 3. 双顶/双底检测 ──
        # 寻找两个相近的高点（双顶）或低点（双底）
        mid = n // 2
        left_max = np.max(high_60[:mid])
        right_max = np.max(high_60[mid:])
        left_min = np.min(low_60[:mid])
        right_min = np.min(low_60[mid:])

        # 双顶：左右高点接近且都在高位
        if abs(left_max - right_max) / max(left_max, right_max) < 0.03 and left_max > np.median(close_60):
            patterns_found.append(f"疑似双顶形态(L:{left_max:.3f}, R:{right_max:.3f})")
            score -= 15

        # 双底：左右低点接近且都在低位
        if abs(left_min - right_min) / max(left_min, right_min) < 0.03 and left_min < np.median(close_60):
            patterns_found.append(f"疑似双底形态(L:{left_min:.3f}, R:{right_min:.3f})")
            score += 15

        # ── 4. 近期K线组合判断（最近3根） ──
        if n >= 3:
            c1, c2, c3 = close_60[-3], close_60[-2], close_60[-1]
            o1, o2, o3 = close_60[-4] if n >= 4 else c1, c1, c2  # approximate opens
            # 启明星（看涨反转）：大阴线+小实体+大阳线
            if (c1 < o1 * 0.97 and abs(c2 - o2) / o2 < 0.01 and c3 > o3 * 1.03):
                patterns_found.append("启明星形态(看涨)")
                score += 10
            # 黄昏星（看跌反转）：大阳线+小实体+大阴线
            elif (c1 > o1 * 1.03 and abs(c2 - o2) / o2 < 0.01 and c3 < o3 * 0.97):
                patterns_found.append("黄昏星形态(看跌)")
                score -= 10
            # 三连阳/三连阴
            ret_3 = (close_60[-1] / close_60[-3] - 1) * 100
            ret_2 = (close_60[-1] / close_60[-2] - 1) * 100
            if ret_3 > 3 and ret_2 > 0:
                patterns_found.append("近3日连续上涨")
                score += 5
            elif ret_3 < -3 and ret_2 < 0:
                patterns_found.append("近3日连续下跌")
                score -= 5

        score = float(np.clip(score, 0, 100))

        # 构建带图表的data_text
        chart_lines = []
        for i in range(n):
            bar = "↑" if close_60[i] > close_60[i - 1] else "↓" if i > 0 else "─"
            chart_lines.append(f"  [{i+1:2d}] 收{close_60[i]:.4f} {bar}")
        chart_str = "\n".join(chart_lines[-20:])  # 最近20根K线

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
