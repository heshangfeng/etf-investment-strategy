# ETF多智能体投研系统 - 设计规格

## 1. 概述

基于现有 `etf-agent.py` 改造为 **LLM多角色专家辩论式** ETF投研系统。

### 核心原则
- **LLM优先，规则兜底**：每个Agent优先用LLM深度分析，LLM失败时fallback到现有规则评分
- **选择性辩论**：仅在Agent间存在显著分歧时才触发辩论，避免冗余Token消耗
- **单文件改造**：所有改动在 `etf-agent.py` 内完成
- **渐进兼容**：现有评分函数保留为底层工具

## 2. 智能体角色定义

### 8个Agent一览

| # | Agent名称 | 角色 | 系统Prompt人格 | 核心数据源 |
|---|-----------|------|---------------|-----------|
| 1 | **宏观分析智能体** `MacroAnalystAgent` | 央行视角宏观分析师 | "你是拥有20年经验的央行宏观分析师，擅长从流动性、政策周期、市场情绪判断整体市场环境" | 大盘成交额(`get_market_total_volume`) |
| 2 | **价值估值智能体** `ValueAnalystAgent` | 格雷厄姆式价值投资者 | "你是信奉格雷厄姆-巴菲特价值投资理念的分析师，专注于估值安全边际和均值回归" | PE/PB百分位(`get_index_val`) |
| 3 | **技术趋势智能体** `TechAnalystAgent` | 图表派技术分析师 | "你是拥有15年经验的技术分析师，擅长趋势识别、动量判断和形态分析" | K线价格(`get_etf_price`)、MA5/MA20、波动率 |
| 4 | **舆情情绪智能体** `SentimentAnalystAgent` | 行为金融学专家 | "你是行为金融学专家，擅长从市场情绪、媒体报道中识别非理性繁荣或恐慌" | 新闻爬虫(`get_professional_news`)、情感分析、行业关键词 |
| 5 | **资金流向智能体** `FundFlowAnalystAgent` | 主力资金追踪者 | "你是专注于资金流分析的市场老手，擅长从资金动向发现主力意图" | 北向资金(`get_north_flow`)、成交量比、折溢价 |
| 6 | **风险管理智能体** `RiskManagerAgent` | 首席风控官 | "你是偏保守的首席风控官，对下行风险极度敏感，擅长识别尾部风险" | 波动率、溢价率、流动性 |
| 7 | **行业纵析智能体** `IndustryAnalystAgent` | 行业研究员 | "你是深耕行业的资深研究员，熟悉产业链结构和政策催化逻辑" | 行业关键词库、ETF名称行业匹配 |
| 8 | **首席决策智能体** `ChiefDecisionAgent` | 投委会主席 | "你是投资委员会主席，需要综合各方观点做出最终判断" | 7份Agent报告 + 辩论记录 |

### Agent共通输出结构

```python
@dataclass
class AgentReport:
    agent_name: str           # 智能体名称
    etf_code: str             # ETF代码
    etf_name: str             # ETF名称
    rating: str               # 评级 (强烈看多/看多/中性/看空/强烈看空)
    score: float              # 0-100 量化评分
    analysis: str             # LLM分析报告 (200-500字)
    key_factors: list[str]    # 关键判断因子列表
    risk_warnings: list[str]  # 风险提示列表
    confidence: float          # 置信度 0-1
    data_summary: dict        # 引用的关键数据摘要
```

### 首席最终输出结构

```python
@dataclass
class FinalResearchReport:
    etf_info: dict                    # ETF基本信息 {code, name, type, index_code}
    macro_context: str                # 宏观背景摘要
    agent_reports: list[AgentReport]  # 7份Agent报告
    debates: list                     # 辩论记录 (可能为空)
    final_rating: str                 # 首席最终评级
    position_suggestion: str          # 仓位建议 ("重仓/中仓/轻仓/观望/回避")
    suggested_position_pct: float     # 建议仓位比例 0-1
    core_logic: str                   # 核心投资逻辑 (200字以内)
    risk_summary: str                 # 综合风险提示
    consensus_level: str              # 共识度 (高度一致/基本一致/存在分歧/严重分歧)
```

## 3. 三段式运行流程

### Phase 1: 独立研究 (所有Agent并行)

```
for each ETF in ETF_POOL:
    create_research_group(etf)
    
    parallel:
        # 宏观（全局，一次运行，所有ETF共享）
        macro_report = MacroAnalystAgent.run()
        
        # 以下7个Agent各自独立分析
        value_report = ValueAnalystAgent.run(etf)
        tech_report = TechAnalystAgent.run(etf)
        sentiment_report = SentimentAnalystAgent.run(etf)
        fundflow_report = FundFlowAnalystAgent.run(etf)
        risk_report = RiskManagerAgent.run(etf)
        industry_report = IndustryAnalystAgent.run(etf)
    
    collect_all_reports()
```

每个Agent内部工作流：
```
1. 调用底层数据函数获取原始数据
2. 构造LLM Prompt: System Prompt(角色) + 数据上下文
3. 调用 OpenAI API
4. 解析响应 → 结构化 AgentReport
5. API失败 → 自动回退到规则评分 → 构造 AgentReport
```

### Phase 2: 矛盾检测 & 选择性辩论

```
# 首席检测分歧
ratings = [agent.rating for agent in all_reports]
unique_ratings = set(ratings)

if len(unique_ratings) == 1:
    # 完全一致 → 跳过辩论
    debates = []
elif score_disagreement_magnitude(ratings) < THRESHOLD:
    # 分歧在可接受范围 → 仅记录分歧，不辩论
    debates = [{"note": "轻微分歧，不影响综合判断"}]
else:
    # 显著分歧 → 触发针对性辩论
    opposing_groups = split_by_sentiment(ratings)
    for group_a, group_b in opposing_groups:
        debate_log = debate_round(group_a, group_b, etf_data)
        debates.append(debate_log)
```

**辩论触发阈值**:
- 评级类别差异 >= 2级（如"强烈看多" vs "中性" 或 "看空" vs "强烈看多"）
- 评分差异 >= 25分（0-100分制）

**辩论流程** (2轮):
```
Round 1: Agent A陈述观点 → Agent B反驳
Round 2: Agent B回应反驳 → Agent A最终陈述
```

### Phase 3: 首席综合决策

```
ChiefDecisionAgent.run(
    agent_reports=all_reports,
    debates=debates,
    etf_info=etf
) → FinalResearchReport

首席Agent的决策逻辑:
1. 阅读所有Agent的分析报告
2. 审阅辩论记录（如有）
3. 对每个Agent的置信度加权
4. 输出最终评级、仓位建议、核心逻辑
```

## 4. LLM调用架构

### 统一调用接口

```python
class BaseLLMAgent:
    """所有LLM Agent的基类"""
    
    ROLE_NAME = "基础分析师"
    SYSTEM_PROMPT = "..."
    
    def analyze(self, etf_code, etf_name, **data_kwargs) -> AgentReport:
        # 1. 获取数据
        raw_data = self._collect_data(**data_kwargs)
        # 2. 构造数据摘要
        data_summary = self._summarize_data(raw_data)
        # 3. LLM分析
        report = self._llm_analyze(etf_code, etf_name, data_summary)
        # 4. LLM失败 → fallback
        if report is None:
            report = self._rule_based_fallback(raw_data)
        return report
    
    def _llm_analyze(self, etf_code, etf_name, data_summary) -> AgentReport | None:
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_prompt(etf_code, etf_name, data_summary)}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # 强制JSON输出
            )
            return self._parse_json_response(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️ {self.ROLE_NAME} LLM分析失败: {e}")
            return None
```

### JSON解析格式

每个LLM Agent统一输出JSON:
```json
{
    "rating": "看多",
    "score": 72,
    "analysis": "基于X数据，该标的正处于...",
    "key_factors": ["因素A", "因素B"],
    "risk_warnings": ["风险点X"],
    "confidence": 0.75
}
```

## 5. 现有代码改造明细

### 保留不动
- `WEIGHT` 权重配置 → 作为规则fallback的参考权重
- `PREMIUM_RISK_THRESHOLD` 等阈值
- `CACHE_*` 全局缓存系统
- `BASE_POS_KEYWORDS` / `INDUSTRY_POS_NEG` 关键词库
- `ETF_POOL` 标的池
- 数据采集函数：`get_etf_price`, `get_index_val`, `get_etf_premium`, `get_north_flow`
- 规则评分函数：`ValueScoreAgent.run`, `BoomScoreAgent.run`, 等（作为fallback）

### 改造
- `DataCollectAgent` → 内部保留，改为被Agent调用的工具函数
- `PublicOpinionAgent` → `SentimentAnalystAgent`，增强为LLM Agent
- `DecisionAgent` → `ChiefDecisionAgent`，重写为LLM综合决策+辩论
- `ReportAgent` → 保留报表功能 + 新增完整投研报告输出
- `MainSchedulerAgent` → 三段式调度

### 新增
- `BaseLLMAgent` 基类
- `MacroAnalystAgent` (新)
- `ValueAnalystAgent` (改造ValueScoreAgent)
- `TechAnalystAgent` (改造TechScoreAgent)
- `FundFlowAnalystAgent` (改造FundScoreAgent)
- `RiskManagerAgent` (改造RiskScoreAgent)
- `IndustryAnalystAgent` (新)
- `AgentReport`, `FinalResearchReport` 数据类
- 辩论引擎函数
- `ResearchReportGenerator` 完整报告输出

## 6. 配置区扩展

```python
# 新增LLM模型配置
LLM_MODEL = "gpt-4o-mini"  # 推荐使用性价比高的模型
LLM_MAX_TOKENS = 1024       # 每个Agent输出的最大token数
LLM_TEMPERATURE = 0.3       # 分析温度（低=严谨）

# 辩论配置
DEBATE_ENABLED = True
DEBATE_ROUNDS = 2
DISAGREEMENT_SCORE_THRESHOLD = 25   # 触发辩论的分数差异阈值
DISAGREEMENT_RATING_GAP = 2         # 触发辩论的评级级差
```

## 7. 输出示例

### 每只ETF最终投研报告格式

```
═══════════════════════════════════════════════════════
 ETF多智能体投研报告
═══════════════════════════════════════════════════════
 标的：510300 沪深300ETF | 宽基
═══════════════════════════════════════════════════════
【宏观背景】
全球流动性偏宽松，A股成交额维持在8000亿+，短期政策窗口期...
═══════════════════════════════════════════════════════
【智能体独立分析】
┌─ 价值估值智能体 ─────────────────────────────┐
│ 评级：看多 (65分) | 置信度: 0.75             │
│ PE(TTM)处于近5年35%分位，估值合理偏低...      │
│ 关键因子：估值安全边际、均值回归空间          │
│ 风险：盈利增速可能不及预期                    │
└──────────────────────────────────────────────┘
┌─ 技术趋势智能体 ─────────────────────────────┐
│ 评级：中性 (52分) | 置信度: 0.65             │
│ 价格位于MA5与MA20之间，短期方向不明...        │
│ ...                                          │
└──────────────────────────────────────────────┘
... 共7份Agent报告 ...

═══════════════════════════════════════════════════════
【辩论记录】
⚠️ 价值估值师(看多) vs 技术分析师(中性) 存在分歧
 → 辩论焦点：估值低位是否能抵消技术面的弱势
 → 结论：首席认为当前估值保护足够，技术面等待确认
═══════════════════════════════════════════════════════
【首席决策】
最终评级：看多 | 建议仓位：25%
共识度：基本一致（1/7轻微分歧）

核心逻辑：
沪深300估值处于历史中低位，虽然技术面暂未确认突破，
但宏观流动性支撑和政策窗口期提供安全垫...

综合风险提示：
1. 关注外部事件冲击
2. 盈利修复节奏不确定性
═══════════════════════════════════════════════════════
```

## 8. 错误处理策略

| 场景 | 处理方式 |
|------|---------|
| LLM API超时/失败 | 自动fallback到规则评分，输出标注"规则评分（LLM不可用）" |
| 某Agent数据源不可用 | 该Agent输出标注数据缺失，评分设为50中性，记录原因 |
| 单个ETF数据异常 | 跳过该ETF，输出错误原因，不影响其他ETF |
| 辩论中LLM失败 | 跳过辩论环节，首席直接综合，标注"辩论环节LLM异常" |

## 9. 与现有系统的兼容性

- 现有评分系统的输出作为"规则评分模式"保留
- 用户可通过配置 `LLM_ENABLED = False` 一键切换回纯规则评分模式
- 数据采集层完全兼容，不修改任何akshare调用
