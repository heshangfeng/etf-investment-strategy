"""
ETF 智能投资分析系统 - 顶层主控调度
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import os
from datetime import datetime

from config import (
    LLM_ENABLED, DEBATE_ENABLED, LLM_MODEL, LLM_BASE_URL,
    QUICK_LLM_MODEL, LLM_CALL_COUNT,
    MAIN_WORKERS, AGENT_WORKERS, ETF_POOL
)
from data import (
    DataCollectAgent, ValueScoreAgent, BoomScoreAgent,
    TechScoreAgent, FundScoreAgent, RiskScoreAgent,
    BacktestAgent, EnhancedBacktestAgent, PublicOpinionAgent
)
from models import AgentReport, FinalResearchReport
from agents import (
    MacroAnalystAgent, MonetaryPolicyAgent, PolicyEventAgent,
    ValueAnalystAgent, TechAnalystAgent, SentimentAnalystAgent,
    FundFlowAnalystAgent, RiskManagerAgent, IndustryAnalystAgent,
    RetailSentimentAgent, CrossMarketAgent, BaseLLMAgent,
    HotMoneyAnalystAgent, UnlockPressureAgent, PatternRecognitionAgent,
    TrendPredictorAgent,
)
from debate import DebateEngine
from decision import ChiefDecisionAgent
from report import ResearchReportGenerator
from review import ReviewManager
from portfolio import PortfolioOptimizer, portfolio_optimize


# ====================== 【顶层主控调度 - 三段式多智能体】 ======================
class MainSchedulerAgent:

    @staticmethod
    def _apply_correlation_constraint(final_reports: list[FinalResearchReport]):
        codes = [fr.etf_info['code'] for fr in final_reports]
        price_data = {}
        for code in codes:
            try:
                df = DataCollectAgent.get_etf_price(code)
                if len(df) >= 60:
                    price_data[code] = df["close"].pct_change().dropna().tail(60).values
            except Exception:
                pass
        if len(price_data) < 2:
            return
        sorted_codes = list(price_data.keys())
        price_matrix = np.array([price_data[c] for c in sorted_codes])
        corr_matrix = np.corrcoef(price_matrix)
        n = len(sorted_codes)
        visited = set()
        report_map = {fr.etf_info['code']: fr for fr in final_reports}
        for i in range(n):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and corr_matrix[i][j] > 0.8:
                    group.append(j)
                    visited.add(j)
            if len(group) > 1:
                group_codes = [sorted_codes[idx] for idx in group]
                group_reports = [report_map[c] for c in group_codes if c in report_map]
                if not group_reports:
                    continue
                total_pos = sum(fr.suggested_position_pct for fr in group_reports)
                if total_pos > 0.3:
                    scale = 0.3 / total_pos
                    for fr in group_reports:
                        fr.suggested_position_pct = round(fr.suggested_position_pct * scale, 4)
                    names = [fr.etf_info['name'] for fr in group_reports]
                    print(f"  🔗 相关性约束: {' ↔ '.join(names)} 高度相关(>{0.8}), 总仓位{total_pos*100:.0f}% → 30%")

    def __init__(self):
        self.macro_agent = MacroAnalystAgent()
        self.monetary_agent = MonetaryPolicyAgent()
        self.policy_agent = PolicyEventAgent()
        self.value_agent = ValueAnalystAgent()
        self.tech_agent = TechAnalystAgent()
        self.sentiment_agent = SentimentAnalystAgent()
        self.fundflow_agent = FundFlowAnalystAgent()
        self.risk_agent = RiskManagerAgent()
        self.industry_agent = IndustryAnalystAgent()
        self.retail_sentiment_agent = RetailSentimentAgent()
        self.cross_market_agent = CrossMarketAgent()
        self.hot_money_agent = HotMoneyAnalystAgent()
        self.unlock_agent = UnlockPressureAgent()
        self.pattern_agent = PatternRecognitionAgent()
        self.trend_predictor_agent = TrendPredictorAgent()
        self.chief_agent = ChiefDecisionAgent()
        self.market_state = "震荡偏强"

    def detect_market_state(self) -> str:
        try:
            volume = DataCollectAgent.get_market_total_volume()

            df_csi = DataCollectAgent.get_etf_price("510300")
            if len(df_csi) < 20:
                return "震荡偏强"
            csi_close = df_csi["close"].iloc[-1]
            csi_ma5 = df_csi["ma5"].iloc[-1]
            csi_ma20 = df_csi["ma20"].iloc[-1]
            if csi_close > csi_ma5 and csi_ma5 > csi_ma20:
                csi_trend = 1
            elif csi_close < csi_ma5 and csi_ma5 < csi_ma20:
                csi_trend = -1
            else:
                csi_trend = 0

            csi_vol = df_csi["volatility"].tail(20).mean()
            vol_high = csi_vol > 0.025

            above_ma20 = 0
            total_checked = 0
            for item in ETF_POOL:
                try:
                    df = DataCollectAgent.get_etf_price(item["code"])
                    if len(df) >= 20 and df["close"].iloc[-1] > df["ma20"].iloc[-1]:
                        above_ma20 += 1
                    total_checked += 1
                except Exception:
                    pass
            breadth = above_ma20 / max(total_checked, 1)

            if volume >= 10000 and csi_trend > 0 and breadth > 0.6:
                state = "强趋势牛"
            elif csi_trend < 0 and breadth < 0.3 and volume < 7000:
                state = "强趋势熊"
            elif volume >= 7000 and csi_trend >= 0 and breadth > 0.4:
                state = "震荡偏强"
            elif volume < 7000 or (csi_trend < 0 and breadth < 0.4):
                state = "震荡偏弱"
            else:
                state = "震荡偏强"

            if vol_high and state in ("震荡偏强",):
                state = "震荡偏弱"
            return state
        except Exception:
            return "震荡偏强"

    def _rank_etf_tiers(self) -> dict[int, list]:
        wide = [item for item in ETF_POOL if item["type"] == "宽基"]
        others = [item for item in ETF_POOL if item["type"] != "宽基"]
        scored = []
        for item in others:
            try:
                df = DataCollectAgent.get_etf_price(item["code"])
                if len(df) < 20:
                    scored.append({**item, "_comp": 0})
                    continue
                c = df["close"].values[-20:]
                v = df["volume"].values[-20:]
                vola = df["volatility"].values[-20:]
                scored.append({**item, "_volume": float(np.mean(v)),
                               "_volatility": float(np.mean(vola)),
                               "_momentum_abs": float(abs(c[-1] / c[0] - 1))})
            except Exception:
                scored.append({**item, "_comp": 0})
        valid = [s for s in scored if "_comp" not in s]
        n = len(valid)
        if n == 0:
            return {1: list(wide), 2: [], 3: []}
        vols = np.array([s["_volume"] for s in valid])
        volas = np.array([s["_volatility"] for s in valid])
        moms = np.array([s["_momentum_abs"] for s in valid])
        r_vol = pd.Series(vols).rank(method="average").values / n
        r_vola = pd.Series(volas).rank(method="average").values / n
        r_mom = pd.Series(moms).rank(method="average").values / n
        for i, s in enumerate(valid):
            s["_comp"] = r_vol[i] * 0.4 + r_vola[i] * 0.3 + r_mom[i] * 0.3
        valid.sort(key=lambda x: x["_comp"], reverse=True)
        clean = [{"code": s["code"], "name": s["name"], "type": s["type"], "index_code": s["index_code"]} for s in valid]
        return {1: ETF_POOL, 2: [], 3: []}

    def _rule_based_research(self, code: str, name: str, typ: str, idx: str,
                              global_max_pos: float, macro_report: AgentReport,
                              market_state: str) -> FinalResearchReport:
        rs_value = ValueScoreAgent.run(idx)
        rs_tech = TechScoreAgent.run(code)
        rs_fund = FundScoreAgent.run(code, idx)
        rs_risk = RiskScoreAgent.run(code)
        reports = [
            self.chief_agent._parse_to_report(code, name, None, rs_value),
            self.chief_agent._parse_to_report(code, name, None, rs_tech),
            self.chief_agent._parse_to_report(code, name, None, rs_fund),
            self.chief_agent._parse_to_report(code, name, None, rs_risk),
        ]
        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
        adjusted = []
        for r in reports:
            s = r.score
            if r.agent_name in ChiefDecisionAgent.REVERSE_AGENTS:
                s = 100 - s
            adjusted.append(s)
        zs = ChiefDecisionAgent._compute_zscore(adjusted)
        ns = [float(np.clip(50 + z * 15, 0, 100)) for z in zs]
        wm = ChiefDecisionAgent.load_agent_weights()
        wv = []
        for r in reports:
            w = wm.get(r.agent_name, 1.0) * (0.5 + r.confidence)
            wv.append(w)
        tw = sum(wv)
        ws_score = sum((w / tw) * s for w, s in zip(wv, ns)) if tw > 0 else float(np.mean(ns))
        ts_mom = ChiefDecisionAgent._time_series_momentum_score(code)
        fs = float(np.clip(ws_score + ts_mom, 0, 100))
        cv = float(np.std(ns)) / max(float(np.mean(ns)), 1)
        consensus = "高度一致" if cv < 0.1 else "基本一致" if cv < 0.2 else "存在分歧" if cv < 0.35 else "严重分歧"
        fr = BaseLLMAgent._score_to_rating(fs)
        pm = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}.get(market_state, 1.0)
        am = global_max_pos * pm
        kp = ChiefDecisionAgent._kelly_position(0.55, 1.5, 1.0, am)
        if fs >= 80:
            op, hd = ("强烈买入", "短期(1-4周)") if consensus in ("高度一致", "基本一致") else ("买入", "中期(1-3月)")
        elif fs >= 65:
            op, hd = "买入", "中期(1-3月)"
        elif fs >= 50:
            op, hd = ("长期持有", "长期(6月+)") if typ == "宽基" else ("持有", "中期(1-3月)")
        elif fs >= 35:
            op, hd = "减持", "短期(1-4周)"
        elif fs >= 20:
            op, hd = "卖出", "短期(1-4周)"
        else:
            op, hd = "强烈卖出", "短期(1-4周)"
        if consensus == "严重分歧":
            if op in ("强烈买入", "买入"):
                op, hd = "持有", "中期(1-3月)"
            elif op in ("长期持有",):
                op, hd = "减持", "短期(1-4周)"
        po = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20, "持有": 0.15, "减持": 0.05, "卖出": 0.0, "强烈卖出": 0.0}
        pp = min(kp, am * po.get(op, 0.1))
        pt = {"强烈买入": "重仓", "买入": "中仓", "长期持有": "长持", "持有": "轻仓", "减持": "减仓", "卖出": "卖出", "强烈卖出": "清仓"}
        ar = list(set([w for r in reports for w in r.risk_warnings]))[:5]
        try:
            dr = DataCollectAgent.get_etf_price(code)
            hv = float(dr["volatility"].rolling(20).mean().iloc[-1])
            vf = max(hv * 100, 1.0)
            sl, tp = round(-max(vf * 2.0, 3.0), 1), round(max(vf * 4.0, 6.0), 1)
        except:
            sl, tp = -5.0, 15.0
        return FinalResearchReport(
            etf_info={"code": code, "name": name, "type": typ, "index_code": idx},
            macro_context=macro_report.analysis[:200], agent_reports=reports, debates=[],
            final_rating=fr, position_suggestion=pt.get(op, "观望"),
            suggested_position_pct=round(pp, 2), final_score=round(fs, 1),
            core_logic=f"【规则评分】z-score加权:{ws_score:.1f}分 | 动量:{ts_mom:+.1f} | 共识:{consensus}",
            risk_summary="; ".join(ar) if ar else "暂无显著风险提示",
            consensus_level=consensus, operation=op, holding_period=hd,
            stop_loss_pct=sl, take_profit_pct=tp, factor_contributions={}
        )

    def run(self):
        print("█"*160)
        print("  ETF 多智能体投研系统 | LLM多角色专家分析 + 辩论 + 首席决策")
        print("█"*160)

        # Phase 0: 宏观分析
        print(f"\n【Phase 0】宏观环境分析...")
        macro_report = self.macro_agent.run()
        print(f"  ✅ 宏观: {macro_report.rating} ({macro_report.score}分) | {macro_report.analysis[:80]}")
        global_max_pos = macro_report.score / 100

        if not LLM_ENABLED:
            print("  ⚙️ LLM开关=OFF，使用规则评分模式")

        # Phase 0.5: 全局政策&流动性分析
        print(f"\n【Phase 0.5】政策与流动性分析...")
        monetary_report = self.monetary_agent.run()
        print(f"  ✅ 货币政策: {monetary_report.rating} ({monetary_report.score}分)")
        policy_report = self.policy_agent.run()
        print(f"  ✅ 政策周期: {policy_report.rating} ({policy_report.score}分)")

        print(f"\n【全局仓位上限】{global_max_pos*100:.0f}%")
        print(f"【外层并发】{MAIN_WORKERS} | 【Agent并发】{AGENT_WORKERS}\n")

        # Phase 0.75: 市场状态检测
        self.market_state = self.detect_market_state()
        print(f"【市场状态】{self.market_state}")

        # Phase 0.6: ETF分层
        tiers = self._rank_etf_tiers()
        print(f"\n【全量分析】{len(tiers[1])} 只 ETF，全部 12-Agent + 辩论")
        print()

        # Phase 1: 按Tier顺序处理
        final_reports = []
        tier_labels = {1: "完整分析", 2: "简化分析", 3: "规则评分"}
        for tn in [1, 2, 3]:
            pool = tiers[tn]
            if not pool:
                continue
            print(f"【Phase 1 - Tier{tn} {tier_labels[tn]}】{len(pool)}只ETF...")
            with ThreadPoolExecutor(max_workers=MAIN_WORKERS) as exec:
                fmap = {
                    exec.submit(self._research_single_etf, item, global_max_pos,
                                macro_report, monetary_report, policy_report,
                                self.market_state, tn): item
                    for item in pool
                }
                for future in as_completed(fmap):
                    item = fmap[future]
                    try:
                        fr = future.result()
                        final_reports.append(fr)
                        print(f"  ✅ T{tn} {item['code']} {item['name']} | {fr.final_rating} | 共识:{fr.consensus_level}")
                    except Exception as e:
                        print(f"  ❌ T{tn} {item['code']} {item['name']}：{str(e)[:100]}")

        # Phase 2.5: 组合约束求解——总仓位不超过100%
        total_pos = sum(fr.suggested_position_pct for fr in final_reports)
        if total_pos > 1.0:
            scale = 1.0 / total_pos
            for fr in final_reports:
                fr.suggested_position_pct = round(fr.suggested_position_pct * scale, 4)
            print(f"\n  🔄 组合约束: 原始总仓位{total_pos*100:.0f}% → 归一化至100%")

        # Phase 2.6: ETF相关性约束
        self._apply_correlation_constraint(final_reports)

        # Phase 2.7: 增强回测与组合汇总
        _enhance_with_backtest(final_reports)

        # Phase 2.8: 组合优化（均值-方差/风险平价）
        _portfolio_optimization_phase(final_reports)

        # Phase 3: 报告输出
        ResearchReportGenerator.generate_full_report(final_reports)

        # 复盘：保存快照+T+1验证
        ReviewManager.process(final_reports)

        actionable = [fr for fr in final_reports if fr.suggested_position_pct > 0]
        print(f"\n【主控Agent】全部标的处理完毕！共 {len(final_reports)} 只ETF")
        print(f"【操作建议】买入/持有: {len(actionable)} 只 | 建议不操作: {len(final_reports)-len(actionable)} 只")

        # Phase 4: 个性化投资组合建议 + 自动化模拟交易
        try:
            from portfolio import advise
            advice = advise(final_reports)
            if advice:
                print(advice)
        except Exception as e:
            print(f"  ⚠️ 投资组合建议不可用: {e}")

        try:
            from autotrade import auto_trade
            trades = auto_trade(final_reports)
            b, s = len(trades["buys"]), len(trades["sells"])
            if b or s:
                print(f"\n  🔄 自动调仓: 买入{b}只, 卖出{s}只")
        except Exception as e:
            print(f"  ⚠️ 自动交易不可用: {e}")
        if actionable:
            print(f"【建议持仓】")
            for fr in sorted(actionable, key=lambda x: x.suggested_position_pct, reverse=True):
                print(f"  {fr.etf_info['name']}({fr.etf_info['code']}): {fr.operation} {fr.suggested_position_pct*100:.1f}% | {fr.final_rating}")

        # Phase 5: 工作流总监审查
        try:
            from director import WorkflowDirector
            findings = WorkflowDirector.review(final_reports)
            WorkflowDirector.print_summary(findings)
        except Exception as e:
            print(f"  ⚠️ 工作流审查不可用: {e}")

    def _research_single_etf(self, item: dict, global_max_pos: float,
                               macro_report: AgentReport,
                               monetary_report: AgentReport,
                               policy_report: AgentReport,
                               market_state: str = "震荡偏强",
                               tier: int = 1) -> FinalResearchReport:
        code, name, typ, idx = item["code"], item["name"], item["type"], item["index_code"]

        if tier == 3:
            return self._rule_based_research(code, name, typ, idx, global_max_pos, macro_report, market_state)

        # Tier 1 & 2: LLM Agent analysis
        if tier == 1:
            fns = [self.value_agent.run, self.tech_agent.run, self.sentiment_agent.run,
                   self.fundflow_agent.run, self.risk_agent.run, self.industry_agent.run,
                   self.retail_sentiment_agent.run, self.cross_market_agent.run,
                   self.hot_money_agent.run, self.unlock_agent.run, self.pattern_agent.run,
                   self.trend_predictor_agent.run]
            fargs = [(code, name, idx), (code, name), (code, name), (code, name, idx),
                     (code, name), (code, name), (code,), (name, code),
                     (code,), (code, name), (code,), (code, name)]
        else:
            fns = [self.value_agent.run, self.tech_agent.run, self.sentiment_agent.run, self.fundflow_agent.run]
            fargs = [(code, name, idx), (code, name), (code, name), (code, name, idx)]

        with ThreadPoolExecutor(max_workers=len(fns)) as agent_exec:
            futures = [agent_exec.submit(fn, *fa) for fn, fa in zip(fns, fargs)]
            reports = [f.result() for f in futures]

        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
            r.data_summary["monetary_context"] = monetary_report.analysis[:150]
            r.data_summary["policy_context"] = policy_report.analysis[:150]

        debates = []
        if tier == 1:
            disagreements = DebateEngine.detect_disagreements(reports)
            if disagreements:
                debates = DebateEngine.hold_debate(disagreements, reports, name, code)

        etf_info = {"code": code, "name": name, "type": typ, "index_code": idx}
        final_report = self.chief_agent.run(reports, debates, global_max_pos, etf_info, market_state)
        final_report.macro_context = macro_report.analysis[:200]
        return final_report


# ====================== 【8. 单标的并行任务入口（旧版-保留供参考）】 ======================
# 注意：此函数引用了旧的 DecisionAgent，仅保留供参考
# 新版使用 MainSchedulerAgent._research_single_etf 替代
def single_etf_all_agents(etf_item: dict, global_max_pos: float) -> dict:
    code = etf_item["code"]
    name = etf_item["name"]
    typ = etf_item["type"]
    idx = etf_item["index_code"]

    with ThreadPoolExecutor(max_workers=AGENT_WORKERS) as agent_exec:
        f_val = agent_exec.submit(ValueScoreAgent.run, idx)
        f_boom = agent_exec.submit(BoomScoreAgent.run, code)
        f_tech = agent_exec.submit(TechScoreAgent.run, code)
        f_fund = agent_exec.submit(FundScoreAgent.run, code, idx)
        f_risk = agent_exec.submit(RiskScoreAgent.run, code)
        f_opinion = agent_exec.submit(PublicOpinionAgent.run, name, code)
        f_bt = agent_exec.submit(EnhancedBacktestAgent.run, code)
        f_premium = agent_exec.submit(DataCollectAgent.get_etf_premium, code)

        s_val = f_val.result()
        s_boom = f_boom.result()
        s_tech = f_tech.result()
        s_fund = f_fund.result()
        s_risk = f_risk.result()
        opinion_res = f_opinion.result()
        bt_data = f_bt.result()
        premium = f_premium.result()

    s_opinion = opinion_res["opinion_score"]
    op_tag = opinion_res["opinion_tag"]
    op_desc = opinion_res["opinion_desc"]
    news = opinion_res["news_content"]
    kw = opinion_res["keywords"]
    trend = opinion_res["opinion_trend"]
    llm_text = opinion_res["llm_explain"]
    warn_flag = opinion_res["warn_alert"]

    total_score, pos_rate, action = DecisionAgent.run(
        global_max_pos, s_val, s_boom, s_tech, s_fund, s_risk, s_opinion, warn_flag
    )

    return {
        "代码": code,
        "名称": name,
        "板块类型": typ,
        "估值分": s_val,
        "景气动量分": s_boom,
        "技术趋势分": s_tech,
        "资金流分": s_fund,
        "风控安全分": s_risk,
        "舆情分数": s_opinion,
        "舆情标签": op_tag,
        "舆情简述": op_desc,
        "舆情趋势": trend,
        "资讯摘要": news,
        "情绪关键词": kw,
        "大模型解读": llm_text,
        "利空告警": warn_flag,
        "综合总分": total_score,
        "折溢价率(%)": round(premium*100,2),
        "历史胜率(%)": bt_data["胜率"],
        "盈亏比": bt_data["盈亏比"],
        "最大回撤(%)": bt_data["最大回撤"],
        "120日收益(%)": bt_data["回测收益"],
        "年化收益(%)": bt_data.get("年化收益", 0),
        "夏普比率": bt_data.get("夏普比率", 0),
        "卡玛比率": bt_data.get("卡玛比率", 0),
        "索提诺比率": bt_data.get("索提诺比率", 0),
        "交易次数": bt_data.get("交易次数", 0),
        "建议仓位": f"{pos_rate*100:.0f}%",
        "操作建议": action
    }


# ====================== 【9. 组合回测与增强回测工具】 ======================
class PortfolioBacktest:
    """Aggregate per-ETF backtests into portfolio-level metrics."""

    @staticmethod
    def run(etf_results: list[dict]) -> dict:
        """Aggregate individual ETF backtests.

        Args:
            etf_results: each has etf_info (code, name, type) and backtest (dict with
                         胜率/盈亏比/最大回撤/回测收益/年化收益/夏普比率 etc.)

        Returns:
            Portfolio-level summary dict.
        """
        if not etf_results:
            return {
                "组合胜率(平均)": 0,
                "组合盈亏比(平均)": 0,
                "组合最大回撤": 0,
                "组合收益": 0,
                "等权年化收益": 0,
                "持仓ETF数": 0,
            }

        count = len(etf_results)
        win_rates = []
        pl_ratios = []
        max_drawdowns = []
        cumulative_rets = []
        annual_rets = []

        for r in etf_results:
            bt = r.get("backtest", {})
            win_rates.append(bt.get("胜率", 0))
            pl_ratios.append(bt.get("盈亏比", 0))
            max_drawdowns.append(bt.get("最大回撤", 0))
            cumulative_rets.append(1 + bt.get("回测收益", 0) / 100)
            annual_rets.append(bt.get("年化收益", 0))

        # 组合胜率 and 盈亏比: equal-weight average
        avg_win_rate = float(np.mean(win_rates))
        avg_pl_ratio = float(np.mean(pl_ratios))

        # 组合最大回撤: worst drawdown across all
        worst_dd = float(np.min(max_drawdowns))

        # 组合收益: equal-weighted cumulative （相乘求等权复利收益）
        avg_cum = float(np.mean(cumulative_rets))
        portfolio_return = (avg_cum - 1) * 100

        # 等权年化收益: average of individual annualized returns
        avg_annual_return = float(np.mean(annual_rets))

        return {
            "组合胜率(平均)": round(avg_win_rate, 2),
            "组合盈亏比(平均)": round(avg_pl_ratio, 2),
            "组合最大回撤": round(worst_dd, 2),
            "组合收益": round(portfolio_return, 2),
            "等权年化收益": round(avg_annual_return, 2),
            "持仓ETF数": count,
        }


def _enhance_with_backtest(final_reports: list) -> None:
    """Run EnhancedBacktestAgent for each ETF and attach results to reports.

    Mutates each FinalResearchReport by adding a 'backtest' attribute.
    Also prints per-ETF backtest metrics inline.
    """
    print(f"\n{'='*80}")
    print("  【增强回测】运行逐ETF回测...")
    print(f"{'='*80}")

    etf_results = []
    for fr in final_reports:
        code = fr.etf_info["code"]
        try:
            bt = EnhancedBacktestAgent.run(code)
            fr.backtest = bt  # attach to report
            win = bt.get("胜率", "N/A")
            pl = bt.get("盈亏比", "N/A")
            ret = bt.get("回测收益", "N/A")
            ann = bt.get("年化收益", "N/A")
            sharpe = bt.get("夏普比率", "N/A")
            trades = bt.get("交易次数", "N/A")
            print(f"  ✅ {code} {fr.etf_info['name']:10s} | "
                  f"胜率:{win:>5}% 盈亏比:{pl:>4} 收益:{ret:>6}% "
                  f"年化:{ann:>6}% 夏普:{sharpe:>4} 交易:{trades}")
            etf_results.append({"etf_info": fr.etf_info, "backtest": bt})
        except Exception as e:
            print(f"  ❌ {code} {fr.etf_info['name']}: 回测失败 — {str(e)[:60]}")

    # Portfolio summary
    if etf_results:
        pf = PortfolioBacktest.run(etf_results)
        print(f"\n{'='*80}")
        print("  【组合回测汇总】")
        print(f"{'='*80}")
        print(f"  持仓ETF数:      {pf['持仓ETF数']}")
        print(f"  组合胜率(平均):  {pf['组合胜率(平均)']}%")
        print(f"  组合盈亏比(平均): {pf['组合盈亏比(平均)']}")
        print(f"  组合最大回撤:    {pf['组合最大回撤']}%")
        print(f"  组合收益:        {pf['组合收益']}%")
        print(f"  等权年化收益:    {pf['等权年化收益']}%")
        print(f"{'='*80}\n")
    else:
        print("  ⚠️  所有ETF回测均失败，跳过组合汇总\n")


def _portfolio_optimization_phase(final_reports: list) -> None:
    """Run portfolio optimization and print comparison with current Kelly weights."""
    active = [fr for fr in final_reports if fr.suggested_position_pct > 0]
    if len(active) < 5:
        return

    opt_result = portfolio_optimize(active)
    if opt_result is None:
        return

    codes = opt_result["codes"]
    opt_weights = opt_result["weights"]

    # Build name lookup
    name_map = {fr.etf_info["code"]: fr.etf_info["name"] for fr in final_reports}

    # Current total position
    current_total = sum(fr.suggested_position_pct for fr in active)
    opt_total = min(opt_weights.sum(), current_total * 1.2)  # cap at +20% of current
    scale = opt_total / max(opt_weights.sum(), 1e-10)
    scaled_weights = opt_weights * scale

    print(f"\n{'='*80}")
    print("  【组合优化】风险平价 / 均值-方差")
    print(f"{'='*80}")
    print(f"  方法: {opt_result['method']}")
    print(f"  参与优化ETF数: {len(codes)}")
    print(f"  当前总仓位: {current_total*100:.1f}% → 优化目标: {opt_total*100:.1f}%")
    print(f"  预期年化收益: {opt_result['expected_return']*100:.1f}%")
    print(f"  预期年化波动: {opt_result['expected_vol']*100:.1f}%")
    print(f"  夏普比率: {opt_result['sharpe_ratio']:.2f}")
    print(f"  分散度: {opt_result['diversification_ratio']:.2f}")

    print(f"\n  调仓建议:")
    for code, w in zip(codes, scaled_weights):
        name = name_map.get(code, code)
        current_w = 0.0
        for fr in active:
            if fr.etf_info["code"] == code:
                current_w = fr.suggested_position_pct
                break
        diff = w - current_w
        sign = "+" if diff >= 0 else ""
        print(f"    {name:12s}: {current_w*100:.1f}% → {w*100:.1f}% ({sign}{diff*100:.1f}%)")

    # ── 写回优化权重到每个ETF的建议仓位 ──
    for code, w in zip(codes, scaled_weights):
        for fr in final_reports:
            if fr.etf_info["code"] == code and fr.suggested_position_pct > 0:
                fr.suggested_position_pct = round(w, 4)
                break
    print(f"  ✅ 组合优化已应用 ({opt_result['method']})")

    print(f"{'='*80}\n")


# ====================== 程序入口 ======================
def main():
    print(f"LLM多智能体模式: {'开启' if LLM_ENABLED else '关闭（规则评分模式）'}")
    print(f"辩论功能: {'开启' if DEBATE_ENABLED else '关闭'}")
    if LLM_ENABLED:
        print(f"深度模型: {LLM_MODEL}")
        print(f"快速模型: {QUICK_LLM_MODEL}")
        print(f"多智能体辩论: {'开启(LLM驱动)' if DEBATE_ENABLED else '关闭'}")
        print(f"评分模式: z-score加权 + 凯利公式仓位 + 动态准确率权重")
    else:
        print("提示：设置 LLM_ENABLED=True 并配置有效API密钥以启用LLM深度分析")
    print()

    scheduler = MainSchedulerAgent()
    scheduler.run()

    if LLM_ENABLED:
        c = LLM_CALL_COUNT
        print(f"\nLLM调用统计: 深度={c['deep']}次 | 快速={c['quick']}次 | 总计={c['total']}次")


if __name__ == "__main__":
    main()
