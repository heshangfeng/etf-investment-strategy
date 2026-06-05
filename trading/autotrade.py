"""
ETF 智能投资分析系统 - 自动化模拟交易引擎

每次分析完成后自动执行：读取建议 → 调仓 → 记录 → 复盘。
"""
import json
import os
import glob
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from core.config import ETF_POOL, FEE_RATE
from trading.portfolio import Portfolio, Holding, Transaction, load, save, _price, PORTFOLIO_FILE

from infra.logger import get_logger; logger = get_logger(__name__)

TRADE_LOG = Path(__file__).resolve().parent.parent / "data" / "trade_log.json"
PERF_LOG = Path(__file__).resolve().parent.parent / "data" / "performance.json"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

# ====================== 【信号可信度过滤器】 ======================
class SignalCaliberFilter:
    """
    信号可信度过滤器——用已有分析数据评估每个拟执行交易的质量。
    低可信信号跳过，等待次日确认。
    不引入硬性规则，全数据驱动。
    
    评估维度：
    1. 评分极端程度：距中性50越远，信号越强
    2. 评分变化幅度：对比昨日，变化越大越有意义
    3. 共识度：Agent间一致程度
    4. Agent方向一致性：多少Agent指向同一方向
    5. 持有天数：刚买就卖的可信度天然低
    6. 全市场情绪：所有ETF平均评分proxy市场状态
    """

    THRESHOLD_BASE = 45  # 基础执行阈值

    @classmethod
    def load_prev_scores(cls) -> dict[str, float]:
        """从最近一次历史快照加载各ETF评分。"""
        files = sorted(glob.glob(str(SNAPSHOT_DIR / "*.json")))
        if len(files) < 2:
            return {}
        try:
            with open(files[-2], "r", encoding="utf-8") as f:
                data = json.load(f)
            return {e["code"]: e["final_score"] for e in data.get("etfs", []) if "final_score" in e}
        except Exception:
            return {}

    @classmethod
    def _calc_agent_alignment(cls, agent_reports: list, operation: str) -> float:
        """计算与操作方向一致的Agent比例。"""
        if not agent_reports:
            return 0.5
        is_bullish_op = operation in ("强烈买入", "买入", "长期持有")
        aligned, total = 0, 0
        for ar in agent_reports:
            if ar.rating in ("强烈看多", "看多"):
                if is_bullish_op:
                    aligned += 1
                total += 1
            elif ar.rating in ("强烈看空", "看空"):
                if not is_bullish_op:
                    aligned += 1
                total += 1
        return aligned / max(total, 1)

    @classmethod
    def _infer_market_sentiment(cls, all_reports: list) -> float:
        """用所有ETF平均评分推断市场状态，值越低市场越弱。"""
        scores = [getattr(fr, 'final_score', 50) for fr in all_reports]
        return float(np.mean(scores)) if scores else 50.0

    @classmethod
    def _score_extremity(cls, new_score: float) -> int:
        extremity = abs(new_score - 50)
        if extremity >= 25:
            return 25
        if extremity >= 15:
            return 15
        if extremity >= 8:
            return 5
        return 0

    @classmethod
    def _score_change(cls, new_score: float, prev_score: float | None, is_sell: bool) -> int:
        if prev_score is None:
            return 0
        delta = new_score - prev_score
        effective_delta = -delta if is_sell else delta
        if effective_delta >= 15:
            return 25
        if effective_delta >= 10:
            return 18
        if effective_delta >= 5:
            return 10
        if effective_delta >= 2:
            return 5
        return 0

    @classmethod
    def _score_consensus(cls, consensus_level: str) -> int:
        return {"高度一致": 15, "基本一致": 8, "存在分歧": 0, "严重分歧": -10}.get(consensus_level, 0)

    @classmethod
    def _score_alignment(cls, agent_reports: list, operation: str) -> int:
        alignment = cls._calc_agent_alignment(agent_reports, operation)
        if alignment >= 0.8:
            return 20
        if alignment >= 0.65:
            return 12
        if alignment >= 0.5:
            return 5
        return -10

    @classmethod
    def _score_holding_days(cls, is_sell: bool, days_held: int) -> int:
        if not is_sell:
            return 0
        if days_held <= 1:
            return -15
        if days_held <= 3:
            return -8
        if days_held <= 10:
            return -3
        return 0

    @classmethod
    def _score_market_sentiment(cls, all_reports: list) -> tuple[float, int]:
        mkt_sentiment = cls._infer_market_sentiment(all_reports)
        if mkt_sentiment < 42:
            return (0.80, 55)
        if mkt_sentiment < 47:
            return (0.90, 50)
        if mkt_sentiment > 58:
            return (1.10, 40)
        return (1.0, 45)

    @classmethod
    def evaluate(
        cls,
        code: str,
        fr: 'FinalResearchReport',
        all_reports: list,
        days_held: int,
        prev_score: float | None,
    ) -> tuple[bool, str, float]:
        """
        评估调仓信号可信度。
        
        Returns:
            should_execute: True=执行调仓, False=跳过待确认
            reason: 原因描述
            conviction: 可信度分数 0-100
        """
        operation = fr.operation
        new_score = fr.final_score
        is_sell = operation in ("卖出", "强烈卖出", "减持")

        ex_score = cls._score_extremity(new_score)
        delta_score = cls._score_change(new_score, prev_score, is_sell)
        cons_score = cls._score_consensus(fr.consensus_level)
        align_score = cls._score_alignment(fr.agent_reports, operation)
        days_penalty = cls._score_holding_days(is_sell, days_held)
        market_mult, threshold = cls._score_market_sentiment(all_reports)

        raw = ex_score + delta_score + cons_score + align_score + days_penalty
        conviction = float(np.clip(raw * market_mult, 0, 100))

        if fr.consensus_level == "严重分歧" and (prev_score is None or abs(new_score - prev_score) < 3):
            return (False, f"严重分歧+评分不变, 可信度{conviction:.0f}", conviction)

        should = conviction >= threshold
        reason = (
            f"可信度{conviction:.0f}≥阈值{threshold}, 执行"
            if should else
            f"可信度{conviction:.0f}<阈值{threshold}, 跳过待确认"
        )
        return (should, reason, conviction)


def _calculate_target_positions(all_reports: list, pf, total_asset) -> dict:
    recs = {}
    for fr in all_reports:
        code = fr.etf_info.get("code", "")
        recs[code] = {
            "name": fr.etf_info.get("name", ""),
            "position": fr.suggested_position_pct,
            "operation": fr.operation,
            "rating": fr.final_rating,
            "score": fr.final_score,
        }
    return recs


def _execute_sells(pf, recs: dict, report_map: dict, prev_scores: dict, all_reports: list, today: str, trades: dict):
    for h in list(pf.holdings):
        r = recs.get(h.code)
        if not r:
            continue
        if h.added == today:
            print(f"  ⏳ T+1限制: {h.name}({h.code}) 今日买入，跳过卖出")
            continue
        fr = report_map.get(h.code)
        if fr:
            days_held = (datetime.now() - datetime.strptime(h.added, "%Y-%m-%d")).days if h.added else 999
            should_trade, reason, conv = SignalCaliberFilter.evaluate(
                h.code, fr, all_reports, days_held, prev_scores.get(h.code))
            if not should_trade:
                print(f"  🚫 跳过卖出 {h.name}({h.code}): {reason}")
                continue
        if r["operation"] in ("卖出", "强烈卖出"):
            price = _price(h.code, h.avg_cost)
            proceeds = h.shares * price
            fee_val = proceeds * FEE_RATE
            pf.cash += (proceeds - fee_val)
            pf.transactions.append(Transaction(
                date=today, code=h.code, name=h.name,
                type="sell", shares=h.shares, price=price, fee=round(fee_val, 2),
            ))
            trades["sells"].append({
                "code": h.code, "name": h.name, "shares": h.shares,
                "price": round(price, 4), "proceeds": round(proceeds, 2),
                "reason": r["operation"],
            })
            pf.holdings.remove(h)
        elif r["operation"] == "减持":
            if r["position"] == 0:
                price = _price(h.code, h.avg_cost)
                proceeds = h.shares * price
                fee_val = proceeds * FEE_RATE
                pf.cash += (proceeds - fee_val)
                pf.transactions.append(Transaction(
                    date=today, code=h.code, name=h.name,
                    type="sell", shares=h.shares, price=price, fee=round(fee_val, 2),
                ))
                trades["sells"].append({
                    "code": h.code, "name": h.name, "shares": h.shares,
                    "price": round(price, 4), "proceeds": round(proceeds, 2),
                    "reason": r["operation"],
                })
                pf.holdings.remove(h)
            else:
                price = _price(h.code, h.avg_cost)
                current_value = h.shares * price
                mkt_val = sum(hh.shares * _price(hh.code, hh.avg_cost) for hh in pf.holdings)
                target_total = pf.cash + mkt_val
                target_value = target_total * r["position"]
                target_shares = int(target_value / price / 100) * 100
                sell_shares = h.shares - target_shares
                if sell_shares >= 100:
                    proceeds = sell_shares * price
                    fee_val = proceeds * FEE_RATE
                    pf.cash += (proceeds - fee_val)
                    h.shares = target_shares
                    pf.transactions.append(Transaction(
                        date=today, code=h.code, name=h.name,
                        type="sell", shares=sell_shares, price=price, fee=round(fee_val, 2),
                    ))
                    trades["sells"].append({
                        "code": h.code, "name": h.name, "shares": sell_shares,
                        "price": round(price, 4), "proceeds": round(proceeds, 2),
                        "reason": "减持",
                    })

    mkt_val = sum(hh.shares * _price(hh.code, hh.avg_cost) for hh in pf.holdings)
    for h in list(pf.holdings):
        r = recs.get(h.code)
        if not r or r["position"] <= 0:
            continue
        if h.added == today:
            continue
        price = _price(h.code, h.avg_cost)
        current_value = h.shares * price
        current_pct = current_value / max(pf.cash + mkt_val, 1)
        target_pct = r["position"]
        if target_pct > 0 and abs(current_pct - target_pct) / target_pct > 0.2:
            target_value = (pf.cash + mkt_val) * target_pct
            delta_value = target_value - current_value
            if abs(delta_value) > 1000:
                shares_delta = int(abs(delta_value) / price / 100) * 100
                if shares_delta >= 100:
                    if delta_value > 0:
                        cost = shares_delta * price
                        fee_val = cost * FEE_RATE
                        total_cost = cost + fee_val
                        if total_cost <= pf.cash:
                            pf.cash -= total_cost
                            h.shares += shares_delta
                            h.avg_cost = round((h.cost_total + cost) / h.shares, 4)
                            pf.transactions.append(Transaction(
                                date=today, code=h.code, name=h.name,
                                type="buy", shares=shares_delta, price=price, fee=round(fee_val, 2),
                            ))
                            trades["buys"].append({
                                "code": h.code, "name": h.name, "shares": shares_delta,
                                "price": round(price, 4), "cost": round(cost, 2),
                                "position_target": target_pct, "reason": "再平衡",
                            })
                    else:
                        proceeds = shares_delta * price
                        fee_val = proceeds * FEE_RATE
                        pf.cash += (proceeds - fee_val)
                        h.shares -= shares_delta
                        pf.transactions.append(Transaction(
                            date=today, code=h.code, name=h.name,
                            type="sell", shares=shares_delta, price=price, fee=round(fee_val, 2),
                        ))
                        trades["sells"].append({
                            "code": h.code, "name": h.name, "shares": shares_delta,
                            "price": round(price, 4), "proceeds": round(proceeds, 2),
                            "reason": "再平衡",
                        })


def _execute_buys(pf, recs: dict, report_map: dict, prev_scores: dict, all_reports: list, today: str, trades: dict):
    buys = {c: r for c, r in recs.items()
            if r["position"] > 0 and c not in {h.code for h in pf.holdings}}
    if buys and pf.cash > 0:
        total_rec_pos = sum(r["position"] for r in buys.values())
        allocations = []
        for code, r in buys.items():
            fr = report_map.get(code)
            if fr:
                should_trade, reason, conv = SignalCaliberFilter.evaluate(
                    code, fr, all_reports, 999, prev_scores.get(code))
                if not should_trade:
                    print(f"  🚫 跳过买入 {r['name']}({code}): {reason}")
                    continue
            alloc_ratio = r["position"] / max(total_rec_pos, 0.01)
            alloc_cash = pf.cash * alloc_ratio
            price = _price(code, 0)
            if price <= 0:
                continue
            shares = int(alloc_cash / price / 100) * 100
            if shares < 100:
                continue
            cost = shares * price
            fee_val = cost * FEE_RATE
            total_cost = cost + fee_val
            allocations.append((code, r, shares, cost, fee_val, total_cost, price))

        total_needed = sum(a[5] for a in allocations)
        if total_needed > pf.cash:
            scale = pf.cash / total_needed
            scaled = []
            for code, r, shares, cost, fee_val, total_cost, price in allocations:
                new_shares = int(int(shares * scale) / 100) * 100
                if new_shares >= 100:
                    new_cost = new_shares * price
                    new_fee = new_cost * FEE_RATE
                    new_total = new_cost + new_fee
                    scaled.append((code, r, new_shares, new_cost, new_fee, new_total, price))
            allocations = scaled

        for code, r, shares, cost, fee_val, total_cost, price in allocations:
            pf.cash -= total_cost
            nm = r["name"]
            pf.holdings.append(Holding(
                code=code, name=nm, shares=shares,
                avg_cost=round(price + fee_val / max(shares, 1), 4), added=today,
            ))
            pf.transactions.append(Transaction(
                date=today, code=code, name=nm,
                type="buy", shares=shares, price=price, fee=round(fee_val, 2),
            ))
            trades["buys"].append({
                "code": code, "name": nm, "shares": shares,
                "price": round(price, 4), "cost": round(cost, 2),
                "position_target": r["position"], "reason": r["operation"],
            })


def auto_trade(all_reports: list, date_str: str = "") -> dict:
    """
    根据分析报告自动调仓。
    
    逻辑:
       强烈买入/买入/长期持有 → 按建议仓位买入（如未持有）
       持有                  → 不动（维持现有）
       减持/卖出/强烈卖出     → 全仓卖出
     
    Args:
        all_reports: FinalResearchReport 列表
        date_str: 交易日期（默认当天）
     
    Returns:
        调仓摘要 dict
    """
    pf = load()
    today = date_str or datetime.now().strftime("%Y-%m-%d")

    prev_scores = SignalCaliberFilter.load_prev_scores()
    report_map = {fr.etf_info.get("code", ""): fr for fr in all_reports}

    trades = {"buys": [], "sells": [], "holds": [], "date": today}

    recs = _calculate_target_positions(all_reports, pf, None)
    _execute_sells(pf, recs, report_map, prev_scores, all_reports, today, trades)
    _execute_buys(pf, recs, report_map, prev_scores, all_reports, today, trades)

    for h in pf.holdings:
        r = recs.get(h.code, {})
        trades["holds"].append({
            "code": h.code, "name": h.name, "shares": h.shares,
            "reason": r.get("operation", "持有"),
        })

    save(pf)
    _log_snapshot(pf, today, trades)
    _log_performance(pf, today)

    return trades


def _log_snapshot(pf: Portfolio, date: str, trades: dict):
    """记录每日资产快照（按日期去重，同一天只保留最后一条）。"""
    snapshots = []
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snapshots = json.load(f)
        except Exception:
            logger.warning("读取交易日志快照失败", exc_info=True)
            pass

    market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    entry = {
        "date": date,
        "cash": round(pf.cash, 2),
        "market_value": round(market_value, 2),
        "total": round(pf.cash + market_value, 2),
        "trades": trades,
    }

    # 去重：同一天已存在则替换，否则追加
    for i, s in enumerate(snapshots):
        if s.get("date") == date:
            snapshots[i] = entry
            break
    else:
        snapshots.append(entry)

    with open(TRADE_LOG, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)


def _log_performance(pf: Portfolio, date: str):
    """记录累计表现。"""
    market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    total = pf.cash + market_value

    perf = {"initial_capital": 670000.0}
    if PERF_LOG.exists():
        try:
            with open(PERF_LOG, "r", encoding="utf-8") as f:
                perf = json.load(f)
        except Exception:
            logger.warning("读取表现日志失败", exc_info=True)
            pass

    # 更新
    if "current_value" not in perf:
        perf["current_value"] = total
    else:
        perf["current_value"] = total

    perf.setdefault("high_water_mark", total)
    perf["high_water_mark"] = max(perf["high_water_mark"], total)

    total_return = (total / perf["initial_capital"] - 1) * 100
    perf["total_return_pct"] = round(total_return, 2)

    max_dd = (total - perf["high_water_mark"]) / perf["high_water_mark"] * 100 if perf["high_water_mark"] > 0 else 0
    perf["max_drawdown_pct"] = round(min(perf.get("max_drawdown_pct", 0), max_dd), 2)

    # 计算夏普（如果有足够数据点）
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snaps = json.load(f) or [{"total": 670000.0}]
            if len(snaps) > 1:
                values = [s.get("total", 670000.0) for s in snaps]
                returns = [(values[i] / values[i - 1] - 1) for i in range(1, len(values))]
                if returns:
                    avg_ret = np.mean(returns) * 252
                    std_ret = np.std(returns) * np.sqrt(252)
                    perf["sharpe_ratio"] = round(avg_ret / max(std_ret, 1e-10), 3)
        except Exception:
            logger.warning("计算夏普比率失败", exc_info=True)
            pass

    # ── 日盈亏计算 ──
    try:
        if TRADE_LOG.exists():
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snaps = json.load(f)
            if len(snaps) >= 2:
                prev_total = snaps[-2].get("total", total)
                daily_pnl = total - prev_total
                daily_pnl_pct = round((daily_pnl / prev_total) * 100, 2) if prev_total > 0 else 0
                perf["daily_pnl"] = round(daily_pnl, 2)
                perf["daily_pnl_pct"] = daily_pnl_pct
            elif len(snaps) == 1:
                init = perf.get("initial_capital", total)
                perf["daily_pnl"] = round(total - init, 2)
                perf["daily_pnl_pct"] = round(perf.get("total_return_pct", 0), 2)
        cumulative_pnl = total - perf.get("initial_capital", total)
        perf["cumulative_pnl"] = round(cumulative_pnl, 2)
    except Exception:
        logger.warning("计算日盈亏失败", exc_info=True)
        pass

    perf["last_updated"] = date
    with open(PERF_LOG, "w", encoding="utf-8") as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)


def show_performance():
    """显示策略整体表现。"""
    if not PERF_LOG.exists():
        print("暂无表现数据")
        return

    with open(PERF_LOG, "r", encoding="utf-8") as f:
        perf = json.load(f)

    print(f"\n{'='*50}")
    print(f"  策略表现总览")
    print(f"{'='*50}")
    print(f"  初始资金: {perf.get('initial_capital', 0):>10.2f}")
    print(f"  当前总值: {perf.get('current_value', 0):>10.2f}")
    print(f"  总收益率: {perf.get('total_return_pct', 0):>+8.2f}%")
    print(f"  最大回撤: {perf.get('max_drawdown_pct', 0):>8.2f}%")
    print(f"  夏普比率: {perf.get('sharpe_ratio', 0):>8.3f}")
    print(f"  更新日期: {perf.get('last_updated', 'N/A')}")

    # 如果有交易日志，显示资产曲线
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snaps = json.load(f)
            if len(snaps) > 1:
                print(f"\n  资产曲线:")
                for s in snaps[-10:]:
                    d = s["date"]
                    v = s["total"]
                    ret = (v / perf.get("initial_capital", 670000) - 1) * 100
                    bar = "█" * max(1, int(abs(ret) / 2))
                    print(f"    {d}: {v:>8.2f}  ({ret:>+6.2f}%) {bar}")
        except Exception:
            logger.warning("显示资产曲线失败", exc_info=True)
            pass

    print(f"{'='*50}\n")


def show_trades():
    """显示历史交易记录。"""
    if not TRADE_LOG.exists():
        print("暂无交易记录")
        return

    with open(TRADE_LOG, "r", encoding="utf-8") as f:
        snaps = json.load(f)

    print(f"\n{'='*75}")
    print(f"  历史交易记录")
    print(f"{'='*75}")
    for snap in snaps:
        trades = snap.get("trades", {})
        date = snap.get("date", "?")
        buys = trades.get("buys", [])
        sells = trades.get("sells", [])
        if buys or sells:
            print(f"\n  [{date}]")
            for t in sells:
                fee = t.get("proceeds", 0) * FEE_RATE
                net = t["proceeds"] - fee
                print(f"    卖出 {t['name']}({t['code']}) {t['shares']}份 @ {t['price']}  "
                      f"金额={t['proceeds']:.0f}  佣金={fee:.1f}  净到账={net:.0f}")
            for t in buys:
                fee = t.get("cost", 0) * FEE_RATE
                net = t["cost"] + fee
                print(f"    买入 {t['name']}({t['code']}) {t['shares']}份 @ {t['price']}  "
                      f"金额={t['cost']:.0f}  佣金={fee:.1f}  实付={net:.0f}")

    print(f"\n{'='*60}\n")


def main():
    """CLI 入口。"""
    import sys
    args = sys.argv[1:]

    if not args:
        # 不带参数：显示表现
        show_performance()
    elif args[0] == "trades":
        show_trades()
    elif args[0] == "perf":
        show_performance()
    else:
        print("用法:")
        print("  python autotrade.py          # 查看策略表现")
        print("  python autotrade.py trades   # 查看历史交易")
        print("  python autotrade.py perf     # 查看表现")


if __name__ == "__main__":
    main()

