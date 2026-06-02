"""
ETF 智能投资分析系统 - 自动化模拟交易引擎

每次分析完成后自动执行：读取建议 → 调仓 → 记录 → 复盘。
"""
import json
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config import ETF_POOL, FEE_RATE
from portfolio import Portfolio, Holding, Transaction, load, save, _price, PORTFOLIO_FILE

TRADE_LOG = Path(__file__).parent / "data" / "trade_log.json"
PERF_LOG = Path(__file__).parent / "data" / "performance.json"


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
    holding_codes = {h.code for h in pf.holdings}

    # 解析建议
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

    trades = {"buys": [], "sells": [], "holds": [], "date": today}
    total_sell_proceeds = 0.0
    total_buy_cost = 0.0

    # ── 1. 先卖（释放现金） ──
    for h in list(pf.holdings):
        r = recs.get(h.code)
        if not r:
            continue
        if r["operation"] in ("卖出", "强烈卖出"):
            # 全仓卖出
            price = _price(h.code, h.avg_cost)
            proceeds = h.shares * price
            fee_val = proceeds * FEE_RATE
            pf.cash += (proceeds - fee_val)
            total_sell_proceeds += proceeds

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
                # 全仓卖出
                price = _price(h.code, h.avg_cost)
                proceeds = h.shares * price
                fee_val = proceeds * FEE_RATE
                pf.cash += (proceeds - fee_val)
                total_sell_proceeds += proceeds

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
                # 减持到目标仓位
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
                    total_sell_proceeds += proceeds
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

    # ── 1.5 再平衡（偏差 > 20% 的持仓调整） ──
    mkt_val = sum(hh.shares * _price(hh.code, hh.avg_cost) for hh in pf.holdings)
    for h in list(pf.holdings):
        r = recs.get(h.code)
        if not r or r["position"] <= 0:
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
                        # 买入补仓
                        cost = shares_delta * price
                        fee_val = cost * FEE_RATE
                        total_cost = cost + fee_val
                        if total_cost <= pf.cash:
                            pf.cash -= total_cost
                            total_buy_cost += cost
                            h.shares += shares_delta
                            h.avg_cost = round((h.cost_total + cost) / h.shares, 4)
                            pf.transactions.append(Transaction(
                                date=today, code=h.code, name=h.name,
                                type="buy", shares=shares_delta, price=price, fee=round(fee_val, 2),
                            ))
                            trades["buys"].append({
                                "code": h.code, "name": h.name, "shares": shares_delta,
                                "price": round(price, 4), "cost": round(cost, 2),
                                "position_target": target_pct,
                                "reason": "再平衡",
                            })
                    else:
                        # 卖出减仓
                        proceeds = shares_delta * price
                        fee_val = proceeds * FEE_RATE
                        pf.cash += (proceeds - fee_val)
                        total_sell_proceeds += proceeds
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

    # ── 2. 再买（分配现金） ──
    buys = {c: r for c, r in recs.items()
            if r["position"] > 0 and c not in {h.code for h in pf.holdings}}

    if buys and pf.cash > 0:
        remaining_cash = pf.cash
        total_rec_pos = sum(r["position"] for r in buys.values())
        for code, r in sorted(buys.items(), key=lambda x: x[1]["position"], reverse=True):
            alloc_ratio = r["position"] / max(total_rec_pos, 0.01)
            alloc_cash = remaining_cash * alloc_ratio

            price = _price(code, 0)
            if price <= 0:
                continue
            shares = int(alloc_cash / price / 100) * 100
            if shares < 100:
                continue

            cost = shares * price
            fee_val = cost * FEE_RATE
            total_cost = cost + fee_val
            if total_cost > remaining_cash:
                shares = int(remaining_cash / price / 100) * 100
                if shares < 100:
                    continue
                cost = shares * price
                fee_val = cost * FEE_RATE
                total_cost = cost + fee_val

            # 买入
            remaining_cash -= total_cost
            pf.cash -= total_cost
            total_buy_cost += cost
            nm = r["name"]
            pf.holdings.append(Holding(
                code=code, name=nm, shares=shares,
                avg_cost=round(price, 4), added=today,
            ))
            pf.transactions.append(Transaction(
                date=today, code=code, name=nm,
                type="buy", shares=shares, price=price, fee=round(fee_val, 2),
            ))
            trades["buys"].append({
                "code": code, "name": nm, "shares": shares,
                "price": round(price, 4), "cost": round(cost, 2),
                "position_target": r["position"],
                "reason": r["operation"],
            })

    # ── 3. 继续持有的 ──
    for h in pf.holdings:
        r = recs.get(h.code, {})
        trades["holds"].append({
            "code": h.code, "name": h.name, "shares": h.shares,
            "reason": r.get("operation", "持有"),
        })

    save(pf)

    # ── 4. 记录每日资产快照 ──
    _log_snapshot(pf, today, trades)
    _log_performance(pf, today)

    return trades


def _log_snapshot(pf: Portfolio, date: str, trades: dict):
    """记录每日资产快照。"""
    snapshots = []
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snapshots = json.load(f)
        except Exception:
            pass

    market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    entry = {
        "date": date,
        "cash": round(pf.cash, 2),
        "market_value": round(market_value, 2),
        "total": round(pf.cash + market_value, 2),
        "trades": trades,
    }
    snapshots.append(entry)

    with open(TRADE_LOG, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)


def _log_performance(pf: Portfolio, date: str):
    """记录累计表现。"""
    market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    total = pf.cash + market_value

    perf = {"initial_capital": 100000.0}
    if PERF_LOG.exists():
        try:
            with open(PERF_LOG, "r", encoding="utf-8") as f:
                perf = json.load(f)
        except Exception:
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
                snaps = json.load(f) or [{"total": 100000.0}]
            if len(snaps) > 1:
                values = [s.get("total", 100000.0) for s in snaps]
                returns = [(values[i] / values[i - 1] - 1) for i in range(1, len(values))]
                if returns:
                    avg_ret = np.mean(returns) * 252
                    std_ret = np.std(returns) * np.sqrt(252)
                    perf["sharpe_ratio"] = round(avg_ret / max(std_ret, 1e-10), 3)
        except Exception:
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
                    ret = (v / 100000 - 1) * 100
                    bar = "█" * max(1, int(abs(ret) / 2))
                    print(f"    {d}: {v:>8.2f}  ({ret:>+6.2f}%) {bar}")
        except Exception:
            pass

    print(f"{'='*50}\n")


def show_trades():
    """显示历史交易记录。"""
    if not TRADE_LOG.exists():
        print("暂无交易记录")
        return

    with open(TRADE_LOG, "r", encoding="utf-8") as f:
        snaps = json.load(f)

    print(f"\n{'='*60}")
    print(f"  历史交易记录")
    print(f"{'='*60}")
    for snap in snaps:
        trades = snap.get("trades", {})
        date = snap.get("date", "?")
        buys = trades.get("buys", [])
        sells = trades.get("sells", [])
        if buys or sells:
            print(f"\n  [{date}]")
            for t in sells:
                print(f"    卖出 {t['name']}({t['code']}) {t['shares']}份 @ {t['price']} = {t['proceeds']:.0f}")
            for t in buys:
                print(f"    买入 {t['name']}({t['code']}) {t['shares']}份 @ {t['price']} = {t['cost']:.0f}")

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
