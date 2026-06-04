"""
ETF 智能投资分析系统 - 晨盘增量更新
8:15 AM 运行：加载昨日快照 + 顶层Agent增量更新 + 规则重评分 → 操盘指导
不调LLM，纯规则评分，轻量快速。
"""
import json
import glob
import os
import numpy as np
from datetime import datetime

from config import SNAPSHOT_DIR
from data import DataCollectAgent
from agents import (
    MacroAnalystAgent, MonetaryPolicyAgent, PolicyEventAgent, CrossMarketAgent,
)
from decision import ChiefDecisionAgent
from scheduler import MainSchedulerAgent


SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), SNAPSHOT_DIR)


def load_latest_snapshot() -> dict | None:
    """加载最新的历史快照"""
    files = sorted(glob.glob(os.path.join(SNAPSHOT_PATH, "*.json")))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def get_snapshot_date(data: dict) -> str:
    return data.get("date", data.get("snapshot_date", "unknown"))


def run_top_level_agents() -> dict:
    """运行宏观/政策/跨市场Agent，返回新的宏观context和global_max_pos"""
    macro_agent = MacroAnalystAgent()
    monetary_agent = MonetaryPolicyAgent()
    policy_agent = PolicyEventAgent()
    cross_market_agent = CrossMarketAgent()

    macro_report = macro_agent.run()
    monetary_report = monetary_agent.run()
    policy_report = policy_agent.run()
    cross_market_report = cross_market_agent.run("沪深300", "510300")

    global_max_pos = macro_report.score / 100
    market_state = MainSchedulerAgent().detect_market_state()
    macro_summary = (macro_report.analysis[:120] if macro_report.analysis else "")

    return {
        "macro_score": macro_report.score,
        "global_max_pos": global_max_pos,
        "market_state": market_state,
        "macro_summary": macro_summary,
    }


def re_score_etfs(snapshot_etfs: list, top: dict) -> list[dict]:
    """对快照中的每只ETF应用新的宏观背景重新评分"""
    results = []
    for etf in snapshot_etfs:
        raw_agents = etf.get("agents", [])
        if not raw_agents:
            continue

        agent_scores = [
            {"name": a["n"], "score": a["sc"], "rating": a["rt"]}
            for a in raw_agents
        ]

        result = ChiefDecisionAgent.compute_rule_score(
            agent_scores=agent_scores,
            etf_code=etf["code"],
            etf_name=etf["name"],
            etf_type=etf.get("type", ""),
            global_max_pos=top["global_max_pos"],
            market_state=top["market_state"],
        )

        old_op = etf.get("operation", "")
        old_score = etf.get("final_score", 0)
        new_op = result["operation"]
        new_score = result["final_score"]

        results.append({
            "code": etf["code"],
            "name": etf["name"],
            "type": etf.get("type", ""),
            "old_operation": old_op,
            "new_operation": new_op,
            "old_score": old_score,
            "new_score": new_score,
            "score_delta": round(new_score - old_score, 1),
            "position_pct": result["position_pct"],
            "consensus": result["consensus"],
            "changed": old_op != new_op,
        })

    return results


def apply_portfolio_constraints(results: list[dict]) -> list[dict]:
    """应用总仓位归一化和相关性约束"""
    total_pos = sum(r["position_pct"] for r in results)
    if total_pos > 1.0:
        scale = 1.0 / total_pos
        for r in results:
            r["position_pct"] = round(r["position_pct"] * scale, 4)

    actionable = [r for r in results if r["position_pct"] > 0]
    if len(actionable) >= 2:
        price_data = {}
        for r in actionable:
            try:
                df = DataCollectAgent.get_etf_price(r["code"])
                if len(df) >= 60:
                    price_data[r["code"]] = df["close"].pct_change().dropna().tail(60).values
            except Exception:
                pass

        if len(price_data) >= 2:
            codes = list(price_data.keys())
            price_matrix = np.array([price_data[c] for c in codes])
            corr_matrix = np.corrcoef(price_matrix)
            n = len(codes)
            visited = set()
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
                    group_codes = [codes[idx] for idx in group]
                    group_pos = sum(r["position_pct"] for r in results if r["code"] in group_codes)
                    if group_pos > 0.3:
                        scale = 0.3 / group_pos
                        for r in results:
                            if r["code"] in group_codes:
                                r["position_pct"] = round(r["position_pct"] * scale, 4)

    return results


def format_guidance(results: list[dict], top: dict, snapshot_date: str) -> str:
    """格式化为推送用的操盘指导文本"""
    today_str = datetime.now().strftime("%m月%d日")
    lines = []
    lines.append(f"🏆 今日操盘指导 · {today_str}")
    lines.append("")
    lines.append(f"【市场状态】{top['market_state']} | 仓位上限 {top['global_max_pos']*100:.0f}%")
    if top["macro_summary"]:
        lines.append(f"宏观: {top['macro_summary']}")
    lines.append("")

    strong_buys = [r for r in results if r["new_operation"] == "强烈买入"]
    if strong_buys:
        lines.append(f"🔥 强烈买入 ({len(strong_buys)})")
        for r in strong_buys:
            lines.append(f"  {r['code']} {r['name']}  仓位{r['position_pct']*100:.0f}%")
        lines.append("")

    buys = [r for r in results if r["new_operation"] == "买入"]
    if buys:
        lines.append(f"📈 买入 ({len(buys)})")
        for r in buys:
            lines.append(f"  {r['code']} {r['name']}  仓位{r['position_pct']*100:.0f}%")
        lines.append("")

    risks = [r for r in results if r["new_operation"] in ("减持", "卖出", "强烈卖出")]
    if risks:
        lines.append("⚠️ 风险信号")
        for r in risks:
            lines.append(f"  {r['code']} {r['name']} → {r['new_operation']} (评分{r['new_score']:.0f}, 共识{r['consensus']})")
        lines.append("")

    changes = [r for r in results if r["changed"]]
    if changes:
        lines.append("📊 变化提示")
        for r in changes[:5]:
            delta_str = f"{r['score_delta']:+.1f}"
            lines.append(f"  {r['code']}: {r['old_operation']}→{r['new_operation']} (评分{r['old_score']:.0f}→{r['new_score']:.0f}, {delta_str})")
        lines.append("")

    total_pos = sum(r["position_pct"] for r in results)
    lines.append(f"【建议总仓位】{total_pos*100:.1f}%")
    lines.append(f"数据基于{snapshot_date}快照 + 今日增量更新")

    return "\n".join(lines)


def main():
    snapshot = load_latest_snapshot()
    if snapshot is None:
        print("尚无历史快照数据，请先运行午后全量分析 (python etf-agent.py)")
        return

    snap_date = get_snapshot_date(snapshot)
    print(f"📂 加载快照: {snap_date}, {len(snapshot.get('etfs', []))} 只ETF")

    try:
        snap_dt = datetime.strptime(snap_date, "%Y%m%d")
        days_old = (datetime.now() - snap_dt).days
        if days_old > 1:
            print(f"⚠️ 警告: 快照日期为{snap_date}({days_old}天前)，数据可能陈旧")
    except Exception:
        pass

    print(f"\n🔄 增量更新: 宏观/政策/跨市场Agent...")
    top = run_top_level_agents()
    print(f"  ✅ 宏观: {top['macro_score']:.0f}分 | 市场状态: {top['market_state']} | 仓位上限: {top['global_max_pos']*100:.0f}%")

    etfs = snapshot.get("etfs", [])
    print(f"\n🔄 规则重评分: {len(etfs)} 只ETF...")
    results = re_score_etfs(etfs, top)

    print(f"\n🔄 组合约束...")
    results = apply_portfolio_constraints(results)

    changed_count = sum(1 for r in results if r["changed"])
    print(f"  ✅ 变化: {changed_count} 只ETF操作建议改变")

    guidance = format_guidance(results, top, snap_date)
    print("\n" + "=" * 60)
    print(guidance)
    print("=" * 60)


if __name__ == "__main__":
    main()
