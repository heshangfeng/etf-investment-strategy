"""
ETF 智能投资分析系统 - 数据类定义
"""
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ====================== 【LLM输出结构化验证模型】 ======================

class LLMOutput(BaseModel):
    """LLM 输出的结构化验证模型。所有Agent统一使用此格式。"""
    rating: str = Field(description="评级：强烈看多/看多/中性/看空/强烈看空")
    score: float = Field(default=50, ge=0, le=100, description="0-100 整数分数")
    analysis: str = Field(default="", description="分析报告（200-400字）")
    key_factors: list[str] = Field(default_factory=list, description="关键因子列表")
    risk_warnings: list[str] = Field(default_factory=list, description="风险点列表")
    confidence: float = Field(default=0.5, ge=0, le=1, description="置信度 0-1")

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        allowed = {"强烈看多", "看多", "中性", "看空", "强烈看空"}
        if v not in allowed:
            return "中性"  # 静默修正
        return v

    @classmethod
    def from_llm_json(cls, raw: dict | None) -> Optional["LLMOutput"]:
        """安全解析LLM输出，缺失字段用默认值填充。"""
        if raw is None or not isinstance(raw, dict):
            return None
        try:
            return cls(**raw)
        except Exception:
            # 字段类型不匹配时暴力兼容
            cleaned = {}
            for k in ("rating", "score", "analysis", "key_factors", "risk_warnings", "confidence"):
                v = raw.get(k)
                if k == "rating" and isinstance(v, str):
                    cleaned[k] = v
                elif k == "score" and isinstance(v, (int, float)):
                    cleaned[k] = float(v)
                elif k == "analysis" and isinstance(v, str):
                    cleaned[k] = v
                elif k == "confidence" and isinstance(v, (int, float)):
                    cleaned[k] = float(v)
                elif k in ("key_factors", "risk_warnings") and isinstance(v, list):
                    cleaned[k] = [str(x) for x in v]
            return cls(**cleaned) if cleaned.get("rating") else None


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
