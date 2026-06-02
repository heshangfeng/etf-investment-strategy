"""
ETF 智能投资分析系统 - 组合优化引擎
均值-方差优化 / 风险平价 / Black-Litterman
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PortfolioResult:
    """组合优化结果"""
    etf_codes: list[str]
    weights: np.ndarray          # 优化后的权重数组
    expected_return: float       # 预期年化收益
    expected_vol: float          # 预期年化波动率
    sharpe_ratio: float          # 夏普比率
    diversification_ratio: float # 分散度
    method: str                  # 所用方法


class PortfolioOptimizer:
    """
    组合优化器。支持 Mean-Variance 和 Risk Parity 两种方法。
    使用 scipy.optimize 进行数值优化。
    """

    @staticmethod
    def mean_variance(codes: list[str], returns: np.ndarray,
                      target_return: float | None = None,
                      risk_free_rate: float = 0.025) -> PortfolioResult:
        """
        均值-方差优化（Markowitz）。

        Args:
            codes: ETF代码列表
            returns: T×N 收益率矩阵 (T个交易日, N只ETF)
            target_return: 目标收益率（None=最大化夏普比率）
            risk_free_rate: 无风险利率

        Returns:
            优化后的组合权重和指标
        """
        import scipy.optimize as opt

        n = returns.shape[1]
        mean_ret = returns.mean(axis=0) * 252  # 年化
        cov = np.cov(returns.T) * 252

        def neg_sharpe(weights):
            port_ret = np.dot(weights, mean_ret)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return -(port_ret - risk_free_rate) / max(port_vol, 1e-10)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 0.3)] * n  # 单只ETF上限30%

        # 等权作为初始值
        x0 = np.array([1.0/n] * n)

        result = opt.minimize(neg_sharpe, x0, method='SLSQP',
                             bounds=bounds, constraints=constraints)

        weights = result.x
        port_ret = np.dot(weights, mean_ret)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (port_ret - risk_free_rate) / max(port_vol, 1e-10)
        div_ratio = np.sum(weights * np.sqrt(np.diag(cov))) / max(port_vol, 1e-10)

        return PortfolioResult(
            etf_codes=codes, weights=weights,
            expected_return=port_ret, expected_vol=port_vol,
            sharpe_ratio=sharpe, diversification_ratio=div_ratio,
            method="mean_variance"
        )

    @staticmethod
    def risk_parity(codes: list[str], returns: np.ndarray) -> PortfolioResult:
        """
        风险平价（Risk Parity / Equal Risk Contribution）。
        每只ETF对组合的风险贡献相等。
        """
        import scipy.optimize as opt

        n = returns.shape[1]
        cov = np.cov(returns.T) * 252
        mean_ret = returns.mean(axis=0) * 252

        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return weights * np.dot(cov, weights) / max(port_vol, 1e-10)

        def risk_parity_obj(weights):
            rc = risk_contribution(weights)
            target = 1.0 / n
            return np.sum((rc - target) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 0.3)] * n
        x0 = np.array([1.0/n] * n)

        result = opt.minimize(risk_parity_obj, x0, method='SLSQP',
                             bounds=bounds, constraints=constraints)

        weights = result.x
        port_ret = np.dot(weights, mean_ret)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (port_ret - 0.025) / max(port_vol, 1e-10)
        div_ratio = np.sum(weights * np.sqrt(np.diag(cov))) / max(port_vol, 1e-10)

        return PortfolioResult(
            etf_codes=codes, weights=weights,
            expected_return=port_ret, expected_vol=port_vol,
            sharpe_ratio=sharpe, diversification_ratio=div_ratio,
            method="risk_parity"
        )


def portfolio_optimize(etf_results: list, method: str = "risk_parity") -> dict | None:
    """
    Convenience wrapper: extract price data from FinalResearchReport list, run optimization.

    Args:
        etf_results: list of FinalResearchReport objects with etf_info['code']
        method: 'mean_variance' or 'risk_parity'

    Returns:
        dict with optimized weights and portfolio stats, or None if insufficient data
    """
    from data import DataCollectAgent

    if len(etf_results) < 5:
        return None

    codes = []
    returns_list = []

    for fr in etf_results:
        code = fr.etf_info['code']
        try:
            df = DataCollectAgent.get_etf_price(code)
            if len(df) < 61:
                continue
            # Last 60 daily returns
            daily_ret = df["close"].pct_change().dropna().tail(60).values
            if len(daily_ret) < 30:
                continue
            codes.append(code)
            returns_list.append(daily_ret[-60:])
        except Exception:
            continue

    if len(codes) < 5:
        return None

    returns = np.column_stack(returns_list)

    if method == "mean_variance":
        result = PortfolioOptimizer.mean_variance(codes, returns)
    else:
        result = PortfolioOptimizer.risk_parity(codes, returns)

    return {
        "codes": result.etf_codes,
        "weights": result.weights,
        "expected_return": result.expected_return,
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
        "diversification_ratio": result.diversification_ratio,
        "method": result.method,
    }
