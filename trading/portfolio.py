"""
ETF 智能投资分析系统 - 实盘投资组合管理

记录真实持仓、买入成本，结合分析报告给出个性化建议。

用法:
  python portfolio.py status              # 查看持仓
  python portfolio.py buy <code> <shares> <price>   # 买入
  python portfolio.py sell <code> <shares>          # 卖出
  python portfolio.py cash <amount>                 # 设置现金
"""
import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from core.config import ETF_POOL

from infra.logger import get_logger; logger = get_logger(__name__)

PORTFOLIO_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"


@dataclass
class Holding:
    code: str
    name: str
    shares: int
    avg_cost: float
    added: str = ""

    @property
    def cost_total(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Transaction:
    date: str
    code: str
    name: str
    type: str
    shares: int
    price: float
    fee: float = 0.0


@dataclass
class Portfolio:
    cash: float = 670000.0
    holdings: list[Holding] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)


def load() -> Portfolio:
    if not PORTFOLIO_FILE.exists():
        return Portfolio()
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Portfolio(
            cash=data.get("cash", 670000.0),
            holdings=[Holding(**h) for h in data.get("holdings", [])],
            transactions=[Transaction(**t) for t in data.get("transactions", [])],
        )
    except Exception:
        return Portfolio()


def save(pf: Portfolio):
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cash": pf.cash,
            "holdings": [asdict(h) for h in pf.holdings],
            "transactions": [asdict(t) for t in pf.transactions],
        }, f, ensure_ascii=False, indent=2)


def _name(code: str) -> str:
    for e in ETF_POOL:
        if e["code"] == code:
            return e["name"]
    return code


def _price(code: str, fallback: float) -> float:
    try:
        from core.data import DataCollectAgent
        df = DataCollectAgent.get_etf_price(code)
        return float(df["close"].iloc[-1])
    except Exception:
        return fallback


def buy(code: str, shares: int, price: float, fee: float = 0.0):
    pf = load()
    nm = _name(code)
    cost = shares * price + fee

    if cost > pf.cash:
        print(f"\U0000274c 现金不足: 需要 {cost:.2f}, 可用 {pf.cash:.2f}")
        return

    existing = [h for h in pf.holdings if h.code == code]
    today = datetime.now().strftime("%Y-%m-%d")

    if existing:
        h = existing[0]
        total_cost = h.cost_total + cost
        total_shares = h.shares + shares
        h.avg_cost = round(total_cost / total_shares, 4)
        h.shares = total_shares
    else:
        pf.holdings.append(Holding(code=code, name=nm, shares=shares,
                                   avg_cost=round(price + fee / max(shares, 1), 4),
                                   added=today))

    pf.cash -= cost
    pf.transactions.append(Transaction(date=today, code=code, name=nm,
                                       type="buy", shares=shares, price=price, fee=fee))
    save(pf)
    print(f"\U00002705 买入 {nm}({code}) {shares}份 @ {price:.4f} | 花费 {cost:.2f} | 现金余 {pf.cash:.2f}")


def sell(code: str, shares: int, price: float | None = None, fee: float = 0.0):
    pf = load()
    nm = _name(code)
    existing = [h for h in pf.holdings if h.code == code]

    if not existing:
        print(f"\U0000274c 未持有 {nm}({code})")
        return

    h = existing[0]
    if shares > h.shares:
        shares = h.shares

    if price is None:
        price = _price(code, h.avg_cost)

    proceeds = shares * price - fee
    today = datetime.now().strftime("%Y-%m-%d")

    if shares >= h.shares:
        pf.holdings.remove(h)
    else:
        h.shares -= shares

    pnl_from_sale = proceeds - shares * h.avg_cost
    pf.cash += proceeds
    pf.transactions.append(Transaction(date=today, code=code, name=nm,
                                       type="sell", shares=shares, price=price, fee=fee))
    save(pf)
    print(f"\U00002705 卖出 {nm}({code}) {shares}份 @ {price:.4f} | 回收 {proceeds:.2f} | 盈亏 {pnl_from_sale:+.2f} | 现金余 {pf.cash:.2f}")


def status():
    pf = load()

    if not pf.holdings:
        print("\n  当前持仓: 空仓")
    else:
        total_value = 0.0
        total_cost = 0.0
        print(f"\n  {'代码':<8} {'名称':<14} {'份额':<8} {'成本价':<10} {'现价':<10} {'市值':<10} {'盈亏':<10}")
        print(f"  {'-'*68}")
        for h in pf.holdings:
            cp = _price(h.code, h.avg_cost)
            mv = h.shares * cp
            ct = h.cost_total
            pnl = mv - ct
            pp = (pnl / ct * 100) if ct > 0 else 0
            total_value += mv
            total_cost += ct
            print(f"  {h.code:<8} {h.name:<14} {h.shares:<8} {h.avg_cost:<10.4f} {cp:<10.4f} {mv:<10.2f} {pnl:<+9.2f} ({pp:<+.1f}%)")

        tp = total_value - total_cost
        tpp = (tp / total_cost * 100) if total_cost > 0 else 0
        print(f"  {'-'*68}")
        print(f"  {'总持仓':>34} {total_value:<10.2f} {tp:<+9.2f} ({tpp:<+.1f}%)")

    cash = pf.cash
    assets = cash + sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    print(f"\n  现金: {cash:>10.2f}")
    print(f"  总资产: {assets:>10.2f}")
    print(f"  仓位: {(1 - cash / max(assets, 1)) * 100:.1f}%")
    print(f"  交易记录: {len(pf.transactions)} 笔")


def advise(all_reports: list) -> str:
    """
    对比持仓 vs 分析报告，生成个性化建议。
    """
    pf = load()
    if not all_reports:
        return ""

    recs = {}
    for fr in all_reports:
        code = fr.etf_info.get("code", "")
        recs[code] = {
            "name": fr.etf_info.get("name", ""),
            "rating": fr.final_rating,
            "score": fr.final_score,
            "position": fr.suggested_position_pct,
            "operation": fr.operation,
        }

    holding_codes = {h.code for h in pf.holdings}

    # 建议买入（推荐>0且没持有）
    to_buy = {c: r for c, r in recs.items() if r["position"] > 0 and c not in holding_codes}
    # 建议卖出（持有但推荐0且操作是减持/卖出）
    to_sell = {}
    to_hold = {}
    for h in pf.holdings:
        r = recs.get(h.code)
        if r and r["position"] == 0 and r["operation"] in ("减持", "卖出", "强烈卖出"):
            to_sell[h.code] = (h, r)
        else:
            to_hold[h.code] = (h, r or {"name": h.name, "operation": "持有", "score": 50, "rating": "未知"})

    lines = [f"\n{'='*60}", f"  \u4e2a\u4eba\u6295\u8d44\u7ec4\u5408\u5efa\u8bae", f"{'='*60}"]

    if to_buy:
        lines.append("\n  \U0001f4c8 \u5efa\u8bae\u4e70\u5165\uff08\u7cfb\u7edf\u63a8\u8350\u4f46\u4f60\u672a\u6301\u6709\uff09:")
        for c, r in sorted(to_buy.items(), key=lambda x: x[1]["position"], reverse=True):
            lines.append(f"    {r['name']}({c}): {r['operation']} {r['position']*100:.0f}% | {r['rating']} ({r['score']})")

    if to_sell:
        lines.append("\n  \U0001f4c9 \u5efa\u8bae\u5356\u51fa\uff08\u4f60\u6301\u6709\u4f46\u7cfb\u7edf\u5efa\u8bae\u51cf\u4ed3\uff09:")
        for c, (h, r) in to_sell.items():
            pnl = ""
            try:
                cp = _price(c, h.avg_cost)
                pnl = f"\u5f53\u524d\u76c8\u4e8f {(cp-h.avg_cost)/h.avg_cost*100:+.1f}%"
            except Exception:
                logger.warning("计算卖出建议盈亏失败", exc_info=True)
                pass
            lines.append(f"    {h.name}({c}): {h.shares}\u4efd \u5747\u4ef7{h.avg_cost:.4f} | {pnl} | \u5efa\u8bae{r['operation']}")

    if to_hold:
        lines.append("\n  \u2705 \u7ee7\u7eed\u6301\u6709:")
        for c, (h, r) in sorted(to_hold.items(), key=lambda x: x[1][1].get("score", 50) if isinstance(x[1][1], dict) else 50, reverse=True):
            pnl = ""
            try:
                cp = _price(c, h.avg_cost)
                pnl = f"\u76c8\u4e8f {(cp-h.avg_cost)/h.avg_cost*100:+.1f}%"
            except Exception:
                logger.warning("计算持有建议盈亏失败", exc_info=True)
                pass
            lines.append(f"    {h.name}({c}): {h.shares}\u4efd \u5747\u4ef7{h.avg_cost:.4f} | {pnl} | {r.get('operation','\u6301\u6709')}")

    cash = pf.cash
    assets = cash + sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    lines.append(f"\n  \u73b0\u91d1: {cash:>10.2f}  \u603b\u8d44\u4ea7: {assets:>10.2f}  \u4ed3\u4f4d: {(1-cash/max(assets,1))*100:.1f}%")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 以下为组合优化引擎（均值-方差 / 风险平价）
# ═══════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    """组合优化结果"""
    etf_codes: list[str]
    weights: np.ndarray
    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    diversification_ratio: float
    method: str


class PortfolioOptimizer:
    """组合优化器。支持 Mean-Variance 和 Risk Parity 两种方法。"""

    @staticmethod
    def mean_variance(codes: list[str], returns: np.ndarray,
                      risk_free_rate: float = 0.025) -> "PortfolioResult":
        import scipy.optimize as opt
        n = returns.shape[1]
        mean_ret = returns.mean(axis=0) * 252
        cov = np.cov(returns.T) * 252

        def neg_sharpe(w):
            pr = np.dot(w, mean_ret)
            pv = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -(pr - risk_free_rate) / max(pv, 1e-10)

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 0.3)] * n
        x0 = np.array([1.0 / n] * n)
        res = opt.minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
        w = res.x
        pr = np.dot(w, mean_ret)
        pv = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (pr - risk_free_rate) / max(pv, 1e-10)
        dr = np.sum(w * np.sqrt(np.diag(cov))) / max(pv, 1e-10)
        return PortfolioResult(codes, w, pr, pv, sharpe, dr, "mean_variance")

    @staticmethod
    def risk_parity(codes: list[str], returns: np.ndarray) -> "PortfolioResult":
        import scipy.optimize as opt
        n = returns.shape[1]
        cov = np.cov(returns.T) * 252
        mean_ret = returns.mean(axis=0) * 252

        def rc(w):
            pv = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return w * np.dot(cov, w) / max(pv, 1e-10)

        def obj(w):
            r = rc(w)
            return np.sum((r - 1.0 / n) ** 2)

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 0.3)] * n
        x0 = np.array([1.0 / n] * n)
        res = opt.minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        w = res.x
        pr = np.dot(w, mean_ret)
        pv = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (pr - 0.025) / max(pv, 1e-10)
        dr = np.sum(w * np.sqrt(np.diag(cov))) / max(pv, 1e-10)
        return PortfolioResult(codes, w, pr, pv, sharpe, dr, "risk_parity")


def portfolio_optimize(etf_results: list, method: str = "risk_parity") -> dict | None:
    """从 FinalResearchReport 提取数据执行组合优化。"""
    if len(etf_results) < 5:
        return None
    from core.data import DataCollectAgent
    codes = []
    rets = []
    for fr in etf_results:
        code = fr.etf_info["code"]
        try:
            df = DataCollectAgent.get_etf_price(code)
            if len(df) < 61:
                continue
            dr = df["close"].pct_change().dropna().tail(60).values
            if len(dr) < 30:
                continue
            codes.append(code)
            rets.append(dr[-60:])
        except Exception:
            continue
    if len(codes) < 5:
        return None
    returns = np.column_stack(rets)
    r = PortfolioOptimizer.mean_variance(codes, returns) if method == "mean_variance" else PortfolioOptimizer.risk_parity(codes, returns)
    return {
        "method": r.method, "codes": r.etf_codes, "weights": r.weights.tolist(),
        "expected_return": round(r.expected_return * 100, 2),
        "expected_vol": round(r.expected_vol * 100, 2),
        "sharpe_ratio": round(r.sharpe_ratio, 3),
    }


def cli():
    args = sys.argv[1:]
    if not args or args[0] == "status":
        status()
    elif args[0] == "buy" and len(args) >= 4:
        try:
            buy(args[1], int(args[2]), float(args[3]), float(args[4]) if len(args) > 4 else 0.0)
        except ValueError:
            print("用法: python portfolio.py buy <code> <shares> <price> [fee]")
    elif args[0] == "sell" and len(args) >= 3:
        try:
            sell(args[1], int(args[2]), float(args[3]) if len(args) > 3 else None, float(args[4]) if len(args) > 4 else 0.0)
        except ValueError:
            print("用法: python portfolio.py sell <code> <shares> [price] [fee]")
    elif args[0] == "cash" and len(args) >= 2:
        try:
            pf = load()
            pf.cash = float(args[1])
            save(pf)
            print(f"\u2705 \u73b0\u91d1\u8bbe\u7f6e\u4e3a {float(args[1]):.2f}")
        except ValueError:
            print("用法: python portfolio.py cash <amount>")
    else:
        print("用法:")
        print("  python portfolio.py status              # 查看持仓")
        print("  python portfolio.py buy <code> <shares> <price> [fee]  # 买入")
        print("  python portfolio.py sell <code> <shares> [price] [fee] # 卖出")
        print("  python portfolio.py cash <amount>       # 设置现金")


if __name__ == "__main__":
    cli()

