"""
ETF 智能投资分析系统 - 数据类定义
"""
from dataclasses import dataclass, field


# ====================== 【数据类定义】 ======================

@dataclass
class AgentReport:
    """单个智能体的分析报告"""
    agent_name: str
    etf_code: str
    etf_name: str
    rating: str                # 强烈看多 / 看多 / 中性 / 看空 / 强烈看空
    score: float               # 0-100
    analysis: str              # LLM分析文本
    key_factors: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.5
    data_summary: dict = field(default_factory=dict)
    source: str = "llm"        # "llm" 或 "rule_fallback"


@dataclass
class FinalResearchReport:
    """单只ETF的完整投研报告"""
    etf_info: dict
    macro_context: str = ""
    agent_reports: list[AgentReport] = field(default_factory=list)
    debates: list = field(default_factory=list)
    final_rating: str = "未评级"
    position_suggestion: str = "观望"
    suggested_position_pct: float = 0.0
    final_score: float = 0.0       # 首席综合评分
    core_logic: str = ""
    risk_summary: str = ""
    consensus_level: str = "未知"
    factor_contributions: dict = field(default_factory=dict)
    operation: str = "观望"        # 操作建议：强烈买入/买入/长期持有/持有/减持/卖出/强烈卖出
    holding_period: str = ""      # 持有周期：短期(1-4周)/中期(1-3月)/长期(6月+)
    stop_loss_pct: float = 0.0    # 建议止损位（负值，如-5.0表示跌5%止损）
    take_profit_pct: float = 0.0  # 建议止盈位（正值，如+15.0表示涨15%止盈）
