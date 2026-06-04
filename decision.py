"""
ETF 智能投资分析系统 - 首席决策智能体
"""
import numpy as np
import json
import os
import math
from datetime import datetime

from config import (
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_ENABLED,
    LLM_API_KEY, LLM_BASE_URL, RATING_ORDER, ETF_POOL
)
from models import AgentReport, FinalResearchReport
from data import DataCollectAgent
from agents import BaseLLMAgent


# ====================== 【首席决策智能体】 ======================
class ChiefDecisionAgent(BaseLLMAgent):
    ROLE_NAME = "首席决策智能体"
    AGENT_TEMPERATURE = 0.15

    # 评分Agent列表（用于动态权重和方向校准）
    SCORING_AGENTS = [
        "价值估值智能体", "技术趋势智能体", "舆情情绪智能体", "资金流向智能体",
        "风险管理智能体", "行业纵析智能体", "零售情绪智能体", "跨市场联动智能体",
        "游资情绪智能体", "解禁压力智能体", "技术形态智能体", "趋势预测智能体",
    ]
    # 风险管理Agent是"反向"的——高分=安全，需要反转评分和评级
    REVERSE_AGENTS = {"风险管理智能体"}
    RATING_REVERSE_MAP = {"强烈看多": "强烈看空", "看多": "看空", "中性": "中性",
                          "看空": "看多", "强烈看空": "强烈看多"}

    SYSTEM_PROMPT = """你是投资委员会主席，需要综合各方观点做出最终判断。
你的职责：
1. 阅读所有智能体的独立分析报告
2. 审阅辩论记录（如有）
3. 综合不同维度的观点，考虑每个Agent的置信度和专业性
4. 对分歧点做出仲裁判断
5. 给出明确的最终评级、仓位建议和核心逻辑

参考决策示例：
示例1：Agent报告=[价值:看空(35分), 技术:看多(70分), 情绪:看多(65分), 资金:中性(50分), 风控:安全(90分)] 辩论=[技术vs价值关于估值分歧]
→ 最终: 看多(62分) 逻辑=技术面强势但估值偏高，折中判断有限看多，仓位中等
示例2：Agent报告=[全部看多, 75-90分] 辩论=无分歧
→ 最终: 强烈看多(85分) 逻辑=多维度共振，高置信度看多，仓位重仓
示例3：Agent报告=[价值:看空(30分), 技术:看空(25分), 情绪:中性(50分), 资金:看空(35分)] 辩论=无
→ 最终: 看空(32分) 逻辑=多维度一致偏弱，回避风险

评级标准：
- 强烈看多 (score>=80)：多项指标共振，核心机会
- 看多 (score>=65)：整体向好，有少量顾虑
- 中性 (score>=45)：多空均衡，等待信号
- 看空 (score>=30)：整体偏弱，谨慎回避
- 强烈看空 (score<30)：多项风险暴露，清仓回避"""

    @staticmethod
    def _calibrate_confidence(reports: list[AgentReport], normalized_scores: list[float]) -> list[float]:
        """
        置信度校准：用 Agent 间一致性和分数极端程度替代 LLM 随口说的置信度。

        公式:
          agreement = 同方向Agent占比（看多/看空方向一致性）
          extremity = abs(score - 50) / 50  （分数越极端越自信）
          calibrated = 0.3 + 0.5 * agreement + 0.2 * extremity

        范围 0.3 ~ 1.0，确保即使分歧大也有基础置信度。
        """
        calibrated = []
        for i, (r, ns) in enumerate(zip(reports, normalized_scores)):
            # 方向一致性：多少个Agent在同一方向（排除自身）
            direction = "多" if ns > 55 else "空" if ns < 45 else "中"
            same_dir = 0
            for j, s in enumerate(normalized_scores):
                if j == i:  # 跳过自身
                    continue
                if ("多" if s > 55 else "空" if s < 45 else "中") == direction:
                    same_dir += 1
            agreement = same_dir / max(len(normalized_scores) - 1, 1)
            extremity = abs(ns - 50) / 50.0
            confidence = 0.3 + 0.5 * agreement + 0.2 * extremity
            calibrated.append(float(np.clip(confidence, 0.3, 1.0)))
        return calibrated

    @staticmethod
    def _load_proxy_weights() -> dict:
        """
        在没有 T+1 复盘数据时，用回测胜率和 Agent 自身一致性自举权重。

        回测胜率从 EnhancedBacktestAgent 获取。
        Agent 自一致性：Agent 是否倾向于频繁更改判断（flip）。
        """
        weights = {name: 1.0 for name in ChiefDecisionAgent.SCORING_AGENTS}

        # 1. 尝试从回测结果加载代理准确率
        try:
            from data import EnhancedBacktestAgent
            # 用几只代表性宽基ETF的平均回测胜率作为代理
            etf_codes = [e["code"] for e in ETF_POOL if e["type"] == "宽基"][:3]
            win_rates = []
            for code in etf_codes:
                bt = EnhancedBacktestAgent.run(code, days=120)
                if bt.get("胜率", 0) > 0:
                    win_rates.append(bt["胜率"] / 100)
            if win_rates:
                avg_win_rate = np.mean(win_rates)
                # 以此为中心，各Agent根据各自特点微调
                from config import WEIGHT
                for name in weights:
                    base = avg_win_rate
                    weights[name] = 0.3 + base * 1.4
        except Exception:
            pass

        # 2. 尝试从 memory 加载 flip 检测降权
        try:
            from memory import MemoryRetriever
            flip_counts = {}
            # 检查全市场所有ETF，统计每Agent的flip频率
            for etf in ETF_POOL:
                memory = MemoryRetriever.retrieve(etf["code"], days=10, top_k=10)
                if memory:
                    # 解析memory文本，统计评级变化次数
                    for line in memory.split("\n"):
                        if "|" in line and "历史分析" not in line:
                            parts = line.split("|")
                            if len(parts) >= 3:
                                agent_name = parts[1].strip()
                                if agent_name in ("首席决策",):
                                    continue
                                if agent_name not in flip_counts:
                                    flip_counts[agent_name] = {"total": 0, "flips": 0, "last_rating": ""}
                                rating = parts[2].strip().split("(")[0].strip()
                                fc = flip_counts[agent_name]
                                if fc["last_rating"] and fc["last_rating"] != rating:
                                    fc["flips"] += 1
                                fc["last_rating"] = rating
                                fc["total"] += 1
            for aname, fc in flip_counts.items():
                if fc["total"] >= 3 and aname in weights:
                    flip_rate = fc["flips"] / max(fc["total"], 1)
                    if flip_rate > 0.4:  # 40%+ flip 率 → 降权15%
                        weights[aname] *= 0.85
                    elif flip_rate > 0.6:  # 60%+ → 降权30%
                        weights[aname] *= 0.7
        except Exception:
            pass

        return weights

    @staticmethod
    def load_agent_weights() -> dict:
        """从ReviewManager历史准确率加载动态权重"""
        weights = {name: 1.0 for name in ChiefDecisionAgent.SCORING_AGENTS}
        try:
            # Lazy import to avoid circular dependency
            from review import ReviewManager
            if os.path.exists(ReviewManager.REVIEW_FILE):
                with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                for aname, data in stats.get("by_agent", {}).items():
                    d = data["directional"]
                    h = data["hits"]
                    if d > 2:  # 至少2次即可（降冷启动门槛）
                        acc = h / d
                        weights[aname] = 0.3 + acc * 1.4  # 0.3~1.7范围，0.5准确率=1.0中性
        except:
            pass
        return weights

    @staticmethod
    def _compute_zscore(scores: list[float]) -> list[float]:
        """z-score标准化"""
        arr = np.array(scores)
        mean, std = np.mean(arr), np.std(arr, ddof=1)
        if std < 1e-6:
            return [0.0] * len(scores)
        return [float((s - mean) / std) for s in scores]

    @staticmethod
    def _sigmoid_penalty(value: float, threshold: float, slope: float = 500) -> float:
        """平滑过渡惩罚——使用sigmoid替代硬切断"""
        return 1.0 / (1.0 + math.exp(-slope * (value - threshold)))

    @staticmethod
    def _kelly_position(win_rate: float, avg_win: float, avg_loss: float, max_pos: float) -> float:
        """简化凯利公式仓位计算"""
        if avg_loss <= 0 or win_rate >= 1:
            return max_pos
        b = avg_win / abs(avg_loss)  # 赔率
        p = win_rate
        q = 1 - p
        if b <= 0:
            return 0
        f_star = (p * b - q) / b
        return float(np.clip(f_star * max_pos, 0.05, max_pos))

    @staticmethod
    def _time_series_momentum_score(etf_code: str) -> float:
        try:
            df = DataCollectAgent.get_etf_price(etf_code)
            if len(df) < 20:
                return 0.0
            close = df["close"].values[-20:]
            x = np.arange(20)
            A = np.vstack([x, np.ones(20)]).T
            slope, intercept = np.linalg.lstsq(A, close, rcond=None)[0]
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            trend_strength = r2 * (1 if slope > 0 else -1) * 4.0

            consistency = 0
            ma20_series = df["close"].rolling(20).mean()
            for i in range(-20, 0):
                if df["close"].iloc[i] > ma20_series.iloc[i]:
                    consistency += 1
                else:
                    consistency -= 1
            consistency_score = consistency / 20.0 * 4.0

            adx_signal = 0.0
            if "high" in df.columns and "low" in df.columns and len(df) >= 35:
                high = df["high"].values[-35:]
                low = df["low"].values[-35:]
                close_a = df["close"].values[-35:]
                tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close_a[:-1]))
                tr = np.maximum(tr, np.abs(low[1:] - close_a[:-1]))
                atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                up_move = high[1:] - high[:-1]
                down_move = low[:-1] - low[1:]
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                di_plus = np.sum(plus_dm[-14:]) / max(atr_14, 1e-10) * 100
                di_minus = np.sum(minus_dm[-14:]) / max(atr_14, 1e-10) * 100
                dx = abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-10) * 100
                if di_plus > di_minus:
                    adx_signal = dx / 100.0 * 2.0
                else:
                    adx_signal = -dx / 100.0 * 2.0

            score = trend_strength + consistency_score + adx_signal
            return float(np.clip(score, -10, 10))
        except Exception:
            return 0.0

    def run(self, reports: list[AgentReport], debates: list[dict],
            global_max_pos: float, etf_info: dict,
            market_state: str = "震荡偏强") -> FinalResearchReport:

        # ── 1. 评分预处理：RiskAgent反转 + z-score标准化 ──
        adjusted_scores = []
        for r in reports:
            score = r.score
            if r.agent_name in self.REVERSE_AGENTS:
                score = 100 - score  # 风控反转：100=安全→0=安全，0=高风险→100=高风险
                # 反转评级显示，避免"强烈看多"对应贡献分数10的矛盾
                if r.rating in self.RATING_REVERSE_MAP:
                    r.rating = self.RATING_REVERSE_MAP[r.rating]
            adjusted_scores.append(score)

        # ── 共识度计算（在 z-score 之前，用原始分数） ──
        raw_std = float(np.std(adjusted_scores))
        raw_mean = float(np.mean(adjusted_scores)) if adjusted_scores else 1
        raw_cv = raw_std / max(raw_mean, 1)
        if raw_cv < 0.15:
            consensus = "高度一致"
        elif raw_cv < 0.25:
            consensus = "基本一致"
        elif raw_cv < 0.40:
            consensus = "存在分歧"
        else:
            consensus = "严重分歧"

        # z-score标准化
        z_scores = self._compute_zscore(adjusted_scores)
        # 转回0-100分制（z-score → 50 + z*15）
        normalized_scores = [float(np.clip(50 + z * 15, 0, 100)) for z in z_scores]

        # ── 2. 置信度校准 + 动态加权平均 ──
        # 校准置信度：基于Agent间一致性和分数极端程度
        calibrated_confidences = self._calibrate_confidence(reports, normalized_scores)

        # 加载权重：优先用 T+1 复盘权重，无数据时用代理权重
        weights = self.load_agent_weights()
        has_real_weights = any(w != 1.0 for w in weights.values())
        if not has_real_weights:
            weights = self._load_proxy_weights()

        weight_values = []
        for r, cc in zip(reports, calibrated_confidences):
            w = weights.get(r.agent_name, 1.0)
            # 使用校准后的置信度替代 raw confidence
            w *= (0.5 + cc)
            weight_values.append(w)

        # 因子贡献追踪：(normalized_score - 50) * weight_factor
        factor_contributions = {}
        for r, ns, wv in zip(reports, normalized_scores, weight_values):
            factor_contributions[r.agent_name] = round((ns - 50) * wv, 2)

        total_w = sum(weight_values)
        if total_w > 0:
            norm_weights = [w / total_w for w in weight_values]
            weighted_score = sum(n * s for n, s in zip(norm_weights, normalized_scores))
        else:
            weighted_score = np.mean(normalized_scores)

        # ── 2.5 时间序列动量因子调整 ──
        ts_momentum = self._time_series_momentum_score(etf_info['code'])
        weighted_score += ts_momentum * 0.35

        # ── 3. 分歧折扣（用共识度调整评分） ──
        if market_state == "强趋势牛":
            discount = 1.0
        elif market_state == "震荡偏强":
            discount = max(1.0 - max(raw_cv - 0.15, 0) * 0.5, 0.90)
        elif market_state == "震荡偏弱":
            discount = max(1.0 - max(raw_cv - 0.12, 0) * 0.6, 0.85)
        else:
            discount = max(1.0 - max(raw_cv - 0.10, 0) * 0.7, 0.80)
        weighted_score *= discount

        final_score = float(np.clip(weighted_score, 0, 100))

        # ── 4. 从ReviewManager加载历史胜率（凯利公式用）──
        win_rate = 0.55  # 默认
        avg_win_ratio = 1.5  # 默认盈亏比
        total_verified = 0
        try:
            # Lazy import to avoid circular dependency
            from review import ReviewManager
            if os.path.exists(ReviewManager.REVIEW_FILE):
                with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                total_verified = stats.get("total_verifications", 0)
                overall_acc = stats.get("overall_accuracy_pct", 55) / 100
                if total_verified > 10:
                    win_rate = overall_acc
        except:
            pass

        # Fallback: use backtest data as proxy win rate
        if win_rate == 0.55 and total_verified <= 10:
            try:
                from data import EnhancedBacktestAgent
                bt_codes = ["159915", "510300", "588000"]  # representative ETFs
                bt_wins = []
                bt_pl = []
                for c in bt_codes:
                    bt = EnhancedBacktestAgent.run(c, days=120)
                    if bt.get("胜率", 0) > 0:
                        bt_wins.append(bt["胜率"] / 100)
                    if bt.get("盈亏比", 0) > 0:
                        bt_pl.append(bt["盈亏比"])
                if bt_wins:
                    win_rate = float(np.mean(bt_wins))
                if bt_pl:
                    avg_win_ratio = float(np.mean(bt_pl))
            except Exception:
                pass

        # ── 5. LLM决策（优先于规则，结果决定最终评分）──
        ratings_list = [r.rating for r in reports]
        reports_text = "\n\n".join([
            f"【{r.agent_name}】评级:{r.rating} 评分:{normalized_scores[i]:.0f} 置信度:{r.confidence}\n分析:{r.analysis[:300]}\n关键因子:{'; '.join(r.key_factors)}\n风险:{'; '.join(r.risk_warnings)}"
            for i, r in enumerate(reports)
        ])
        debates_text = ""
        if debates:
            for d in debates:
                debates_text += f"\n分歧: {d['topic']}\n"
                if d.get("winner"):
                    debates_text += f"  仲裁胜方: {d['winner']}\n"
                    debates_text += f"  理由: {d.get('reasoning', '')[:300]}\n"
                    if "score_adjustment" in d:
                        debates_text += f"  评分调整: {d['score_adjustment']:+d}分\n"
                else:
                    debates_text += f"  裁决不可用\n"

        # Note: score adjustments from debates are now applied in scheduler.py
        # before ChiefDecisionAgent.run() — they directly modify AgentReport.score,
        # so normalized_scores already reflect the debate results. No need to re-apply here.

        data_text = f"【ETF信息】{etf_info['name']}({etf_info['code']})\n\n【智能体报告】\n{reports_text}\n\n【仲裁记录】\n{debates_text}\n\n【全局仓位上限】{global_max_pos*100:.0f}%"
        llm_out = self._call_llm(self.SYSTEM_PROMPT,
                                  self._build_user_prompt(etf_info['name'], etf_info['code'], data_text))

        if llm_out:
            llm_final_rating = llm_out.rating
            llm_final_score = float(np.clip(llm_out.score, 0, 100))
            llm_core_logic = llm_out.analysis
            # LLM结果优于规则时采用
            if abs(llm_final_score - 50) > abs(final_score - 50):
                final_score = llm_final_score
                final_rating = llm_final_rating
                core_logic = llm_core_logic
            else:
                final_rating = self._score_to_rating(final_score)
                core_logic = f"【规则综合】z-score加权:{final_score:.1f}分 | 共识:{consensus}"
        else:
            final_rating = self._score_to_rating(final_score)
            core_logic = f"【规则综合】z-score加权:{final_score:.1f}分 | Agent数:{len(reports)} | 共识:{consensus}"

        # ── 6. 操作建议映射（使用最终 final_score，确保与评级一致）──
        etf_type = etf_info.get("type", "")
        if final_score >= 80:
            if consensus in ("高度一致", "基本一致"):
                operation, holding = "强烈买入", "短期(1-4周)"
            else:
                operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 65:
            operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 50:
            operation, holding = "持有", "中期(1-3月)"
        elif final_score >= 35:
            operation, holding = "减持", "短期(1-4周)"
        elif final_score >= 20:
            operation, holding = "卖出", "短期(1-4周)"
        else:
            operation, holding = "强烈卖出", "短期(1-4周)"

        # 严重分歧修正
        if consensus == "严重分歧":
            if operation in ("强烈买入", "买入"):
                operation, holding = "持有", "中期(1-3月)"
            elif operation in ("长期持有",):
                operation, holding = "减持", "短期(1-4周)"

        # ── 7. 凯利公式仓位（根据操作建议计算）──
        market_pos_mult = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}
        pos_mult = market_pos_mult.get(market_state, 1.0)
        adjusted_max_pos = global_max_pos * pos_mult
        kelly_pos = self._kelly_position(win_rate, avg_win_ratio, 1.0, adjusted_max_pos)
        pos_map = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20,
                    "持有": 0.0, "减持": 0.0, "卖出": 0.0, "强烈卖出": 0.0}
        baseline_pos = adjusted_max_pos * pos_map.get(operation, 0.1)
        pos_pct = min(kelly_pos, baseline_pos)
        pos_text_map = {"强烈买入": "重仓", "买入": "中仓", "长期持有": "长持",
                        "持有": "轻仓", "减持": "减仓", "卖出": "卖出", "强烈卖出": "清仓"}

        all_risks = list(set([w for r in reports for w in r.risk_warnings]))[:5]

        # ── 8. 止损止盈计算 ──
        try:
            df = DataCollectAgent.get_etf_price(etf_info['code'])
            hist_vol = float(df["volatility"].rolling(20).mean().iloc[-1])
            vol_factor = max(hist_vol * 100, 1.0)
            stop_loss_pct = round(-max(vol_factor * 2.0, 3.0), 1)
            take_profit_pct = round(max(vol_factor * 4.0, 6.0), 1)
        except:
            stop_loss_pct = -5.0
            take_profit_pct = 15.0

        return FinalResearchReport(
            etf_info=etf_info,
            macro_context="",
            agent_reports=reports,
            debates=debates,
            final_rating=final_rating,
            position_suggestion=pos_text_map.get(operation, "观望"),
            suggested_position_pct=round(pos_pct, 2),
            final_score=round(final_score, 1),
            core_logic=core_logic,
            risk_summary="; ".join(all_risks) if all_risks else "暂无显著风险提示",
            consensus_level=consensus,
            operation=operation,
            holding_period=holding,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            factor_contributions=factor_contributions
        )

    @staticmethod
    def compute_rule_score(
        agent_scores: list[dict],
        etf_code: str,
        etf_name: str,
        etf_type: str,
        global_max_pos: float,
        market_state: str = "震荡偏强",
    ) -> dict:
        adjusted_scores = []
        agent_names = []
        for a in agent_scores:
            name = a["name"]
            score = a["score"]
            if name in ChiefDecisionAgent.REVERSE_AGENTS:
                score = 100 - score
            adjusted_scores.append(score)
            agent_names.append(name)

        raw_std = float(np.std(adjusted_scores))
        raw_mean = float(np.mean(adjusted_scores)) if adjusted_scores else 1
        raw_cv = raw_std / max(raw_mean, 1)
        if raw_cv < 0.15:
            consensus = "高度一致"
        elif raw_cv < 0.25:
            consensus = "基本一致"
        elif raw_cv < 0.40:
            consensus = "存在分歧"
        else:
            consensus = "严重分歧"

        z_scores = ChiefDecisionAgent._compute_zscore(adjusted_scores)
        normalized_scores = [float(np.clip(50 + z * 15, 0, 100)) for z in z_scores]

        calibrated_confidences = []
        for i, ns in enumerate(normalized_scores):
            direction = "多" if ns > 55 else "空" if ns < 45 else "中"
            same_dir = 0
            for j, s in enumerate(normalized_scores):
                if j == i:
                    continue
                if ("多" if s > 55 else "空" if s < 45 else "中") == direction:
                    same_dir += 1
            agreement = same_dir / max(len(normalized_scores) - 1, 1)
            extremity = abs(ns - 50) / 50.0
            confidence = 0.3 + 0.5 * agreement + 0.2 * extremity
            calibrated_confidences.append(float(np.clip(confidence, 0.3, 1.0)))

        weights = ChiefDecisionAgent.load_agent_weights()
        has_real_weights = any(w != 1.0 for w in weights.values())
        if not has_real_weights:
            weights = ChiefDecisionAgent._load_proxy_weights()

        weight_values = []
        for name, cc in zip(agent_names, calibrated_confidences):
            w = weights.get(name, 1.0)
            w *= (0.5 + cc)
            weight_values.append(w)

        total_w = sum(weight_values)
        if total_w > 0:
            norm_weights = [w / total_w for w in weight_values]
            weighted_score = sum(n * s for n, s in zip(norm_weights, normalized_scores))
        else:
            weighted_score = float(np.mean(normalized_scores))

        ts_momentum = ChiefDecisionAgent._time_series_momentum_score(etf_code)
        weighted_score += ts_momentum * 0.35

        if market_state == "强趋势牛":
            discount = 1.0
        elif market_state == "震荡偏强":
            discount = max(1.0 - max(raw_cv - 0.15, 0) * 0.5, 0.90)
        elif market_state == "震荡偏弱":
            discount = max(1.0 - max(raw_cv - 0.12, 0) * 0.6, 0.85)
        else:
            discount = max(1.0 - max(raw_cv - 0.10, 0) * 0.7, 0.80)
        weighted_score *= discount

        final_score = float(np.clip(weighted_score, 0, 100))
        final_rating = ChiefDecisionAgent._score_to_rating(final_score)

        if final_score >= 80:
            if consensus in ("高度一致", "基本一致"):
                operation, holding = "强烈买入", "短期(1-4周)"
            else:
                operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 65:
            operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 50:
            operation, holding = "持有", "中期(1-3月)"
        elif final_score >= 35:
            operation, holding = "减持", "短期(1-4周)"
        elif final_score >= 20:
            operation, holding = "卖出", "短期(1-4周)"
        else:
            operation, holding = "强烈卖出", "短期(1-4周)"

        if consensus == "严重分歧":
            if operation in ("强烈买入", "买入"):
                operation, holding = "持有", "中期(1-3月)"
            elif operation in ("长期持有",):
                operation, holding = "减持", "短期(1-4周)"

        win_rate = 0.55
        avg_win_ratio = 1.5
        total_verified = 0
        try:
            from review import ReviewManager
            if os.path.exists(ReviewManager.REVIEW_FILE):
                with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                total_verified = stats.get("total_verifications", 0)
                overall_acc = stats.get("overall_accuracy_pct", 55) / 100
                if total_verified > 10:
                    win_rate = overall_acc
        except:
            pass

        if win_rate == 0.55 and total_verified <= 10:
            try:
                from data import EnhancedBacktestAgent
                bt_codes = ["159915", "510300", "588000"]
                bt_wins = []
                bt_pl = []
                for c in bt_codes:
                    bt = EnhancedBacktestAgent.run(c, days=120)
                    if bt.get("胜率", 0) > 0:
                        bt_wins.append(bt["胜率"] / 100)
                    if bt.get("盈亏比", 0) > 0:
                        bt_pl.append(bt["盈亏比"])
                if bt_wins:
                    win_rate = float(np.mean(bt_wins))
                if bt_pl:
                    avg_win_ratio = float(np.mean(bt_pl))
            except Exception:
                pass

        market_pos_mult = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}
        pos_mult = market_pos_mult.get(market_state, 1.0)
        adjusted_max_pos = global_max_pos * pos_mult
        kelly_pos = ChiefDecisionAgent._kelly_position(win_rate, avg_win_ratio, 1.0, adjusted_max_pos)
        pos_map = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20,
                    "持有": 0.0, "减持": 0.0, "卖出": 0.0, "强烈卖出": 0.0}
        baseline_pos = adjusted_max_pos * pos_map.get(operation, 0.1)
        pos_pct = min(kelly_pos, baseline_pos)

        try:
            from data import DataCollectAgent
            df = DataCollectAgent.get_etf_price(etf_code)
            hist_vol = float(df["volatility"].rolling(20).mean().iloc[-1])
            vol_factor = max(hist_vol * 100, 1.0)
            stop_loss_pct = round(-max(vol_factor * 2.0, 3.0), 1)
            take_profit_pct = round(max(vol_factor * 4.0, 6.0), 1)
        except:
            stop_loss_pct = -5.0
            take_profit_pct = 15.0

        return {
            "final_score": round(final_score, 1),
            "final_rating": final_rating,
            "operation": operation,
            "holding_period": holding,
            "position_pct": round(pos_pct, 2),
            "consensus": consensus,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
