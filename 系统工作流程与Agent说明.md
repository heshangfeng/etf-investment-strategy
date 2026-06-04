# ETF 智能投研系统 — 工作流程与 Agent 说明

> 本文档描述系统整体架构、各 Agent 职责、完整执行流程以及关键设计决策。
> 当系统升级或新增 Agent 时，请同步更新本文档。

---

## 一、系统定位

基于 **akshare** 实时数据的 ETF 多智能体投研系统，采用 **LLM 多角色专家辩论式** 架构。

每次运行产出 40 只 ETF 的完整投研报告（含每 Agent 分析原文 + 辩论记录 + 首席决策），以及组合优化建议、自动化调仓指令。

---

## 二、架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   Phase 0: 宏观分析                         │
│  MacroAnalystAgent → 市场状态检测 → 全局仓位上限            │
│  MonetaryPolicyAgent → 货币/流动性环境                      │
│  PolicyEventAgent → 政策周期/会议窗口                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ (全局上下文注入每个 Agent)
┌──────────────────────▼──────────────────────────────────────┐
│               Phase 1: 独立研究（12 Agent 并行）             │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐        │
│   │价值  ││技术  ││情绪  ││资金  ││风控  ││行业  │  ...    │
│   │估值  ││趋势  ││舆情  ││流向  ││管理  ││纵析  │         │
│   └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘        │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐        │
│   │零售  ││跨市场││游资  ││解禁  ││技术  ││趋势  │         │
│   │情绪  ││联动  ││情绪  ││压力  ││形态  ││预测  │         │
│   └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘        │
└──────────────────────┬──────────────────────────────────────┘
                       │ (矛盾检测)
┌──────────────────────▼──────────────────────────────────────┐
│            Phase 2: 矛盾检测 → 选择性辩论                    │
│  DebateEngine → 发现 Agent 间分歧 → LLM 仲裁裁决           │
└──────────────────────┬──────────────────────────────────────┘
                       │ (报告 + 辩论记录)
┌──────────────────────▼──────────────────────────────────────┐
│              Phase 3: 首席决策                               │
│  ChiefDecisionAgent → 综合评分 + 凯利仓位 + 操作建议        │
│  (z-score加权 + 置信度校准 + 历史权重 + 动量调整)          │
└──────────────────────┬──────────────────────────────────────┘
                       │ (FinalResearchReport)
┌──────────────────────▼──────────────────────────────────────┐
│           Phase 4: 投后处理                                  │
│  组合约束(归一化) → 相关性约束 → 增强回测                   │
│  → 组合优化(风险平价/均值-方差) → 报告输出                  │
│  → 复盘(T+1验证) → 个性化建议 → 自动调仓                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│           Phase 5: 工作流总监审查                            │
│  WorkflowDirector → 10维度质量审查 + 累积优化建议           │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、Agent 详细介绍

### 3.1 宏观层 Agent（Phase 0）

| Agent | 类名 | 温度 | 模型 | 职责 |
|-------|------|------|------|------|
| **宏观分析** | `MacroAnalystAgent` | 0.15 | 深度模型 | 全市场成交量+沪深300估值+PMI+VIX |
| **货币政策** | `MonetaryPolicyAgent` | 0.15 | 深度模型 | LPR/汇率/M2/中美利差/两融 |
| **政策事件** | `PolicyEventAgent` | 0.60 | 深度模型 | 政治局会议/两会/中央经济工作会议窗口 |

**特点：**
- 使用深度模型（`LLM_MODEL`），承担复杂推理
- 全局运行一次，结果注入每个 ETF 的独立分析
- `MacroAnalystAgent` 同时产出 `global_max_pos`（全局仓位上限）
- `PolicyEventAgent` 自带季节效应判断（春季躁动、中报季等）

### 3.2 基本面 Agent（Phase 1 — 独立研究）

| Agent | 类名 | 文件 | 温度 | 模型 | 分析维度 | 关键数据 |
|-------|------|------|------|------|----------|----------|
| **价值估值** | `ValueAnalystAgent` | `fundamental.py` | 0.3 | 快速 | PE/PB 百分位 | 指数估值 |
| **技术趋势** | `TechAnalystAgent` | `fundamental.py` | 0.3 | 快速 | 均线/RSI/MACD/布林带 | 日K线 |
| **舆情情绪** | `SentimentAnalystAgent` | `fundamental.py` | 0.5 | 快速 | 新闻情绪/舆情趋势 | 专业新闻 |
| **资金流向** | `FundFlowAnalystAgent` | `fundamental.py` | 0.5 | 快速 | 北向资金/量比/折溢价 | 北向/成交量 |
| **风险管理** | `RiskManagerAgent` | `fundamental.py` | 0.15 | 快速 | 溢价/波动率/流动性/VIX | 风控阈值 |
| **行业纵析** | `IndustryAnalystAgent` | `fundamental.py` | 0.4 | 快速 | 行业轮动/相对强度 | 板块资金 |

**共同模式：** 每个 Agent 先**规则评分**（确定性的数学计算），再调用 **LLM 分析**（用同一数据做深度解读），LLM 失败时回退到规则评分。

### 3.3 市场情绪 Agent（Phase 1 — 独立研究）

| Agent | 类名 | 文件 | 温度 | 模型 | 分析维度 |
|-------|------|------|------|------|----------|
| **零售情绪** | `RetailSentimentAgent` | `market.py` | 0.5 | 快速 | 量比/52周价格位置/两融/换手率 |
| **跨市场联动** | `CrossMarketAgent` | `market.py` | 0.6 | 快速 | USDCNY/美债/中美利差/巴菲特指数/基差 |
| **游资情绪** | `HotMoneyAnalystAgent` | `market.py` | 0.4 | 快速 | 涨停数/连板高度/行业分布（反向指标） |
| **解禁压力** | `UnlockPressureAgent` | `market.py` | 0.3 | 快速 | 未来30天解禁市值/解禁日期分布 |
| **技术形态** | `PatternRecognitionAgent` | `market.py` | 0.4 | 快速 | 趋势线/支撑阻力/双顶底/K线组合 |
| **趋势预测** | `TrendPredictorAgent` | `market.py` | 0.3 | 快速 | 短期/中期动量+均线排列+波动率预测 |

**特点：**
- `HotMoneyAnalystAgent` 和 `UnlockPressureAgent` 使用类级别缓存（`_zt_cache` / `_unlock_cache`），避免重复抓取
- `CrossMarketAgent` 采集全球信号但**不依赖 ETF 代码**，个股差异小，主要提供全局背景
- `PatternRecognitionAgent` 含完整的线性回归趋势检测 + 局部极值支撑/阻力计算

### 3.4 辩论引擎（Phase 2）

| 组件 | 类名 | 文件 | 功能 |
|------|------|------|------|
| **仲裁引擎** | `DebateEngine` | `debate.py` | 检测分歧 → 调用 LLM 裁决 → 评分调整 |

**分歧检测规则：**
- 评分差异 ≥ 18 分，或
- 评级级差 ≥ 2 档（如"看多"vs"看空"）

**仲裁流程：**
1. `detect_disagreements()` — 12 Agent 两两比较，找出分歧对
2. `hold_debate()` — 取前 3 个分歧，每对用 LLM 裁决
3. 裁决输出包含胜者、评分配置、分析缺陷
4. 结果传入 `ChiefDecisionAgent`

**裁决格式：**
```json
{"winner": "A/B/折中", "reasoning": "...", "score_adjustment_for_winner": +5}
```

### 3.5 首席决策 Agent（Phase 3）

| 类名 | 文件 | 温度 | 模型 | 职责 |
|------|------|------|------|------|
| `ChiefDecisionAgent` | `decision.py` | 0.15 | 深度模型 | 综合12 Agent + 辩论，产出最终评级/仓位/操作 |

**决策流程（7步）：**

```
1. 评分预处理：RiskAgent 反转（安全=高分→需反转）
2. z-score 标准化(50 + z*15) + 共识度计算(变异系数)
3. 置信度校准(一致性+极端程度) + 动态加权平均
4. 分歧折扣(共识度越低折扣越大)
5. 时间序列动量调整(趋势斜率+R²+ADX)
6. LLM 覆盖(LLM判断优于规则时采用LLM)
7. 操作建议映射 + 凯利公式仓位 + 止损止盈
```

**权重来源（优先级）：**
1. T+1 复盘准确率（`ReviewManager` 累积数据）
2. 回测胜率代理权重（`EnhancedBacktestAgent`）
3. Agent flip 检测降权（`MemoryRetriever`）

**操作建议映射表：**
```
≥80分 → 强烈买入(短期)
≥65分 → 买入(中期)
≥50分 → 长期持有(宽基)/持有(其他)
≥35分 → 减持(短期)
≥20分 → 卖出(短期)
<20分 → 强烈卖出(短期)
```

### 3.6 工作流总监（Phase 5）

| 类名 | 文件 | 功能 |
|------|------|------|
| `WorkflowDirector` | `director.py` | 10 维度质量审查 + 累积优化建议 |

**审查维度：**

| # | 维度 | 方法 | 触发条件 |
|---|------|------|----------|
| 1 | Agent 覆盖完整性 | `review()` | Agent < 12 只 |
| 2 | 分析质量 | `review()` | 字数 < 50 或置信度 < 0.2 |
| 3 | 数据新鲜度 | `review()` | 含"暂无/无数据"关键词 |
| 4 | 评级偏斜 | `review()` | 同一评级 > 80% |
| 5 | 历史准确率 | `_check_historical_accuracy()` | 近 3 次下降 > 5% 或 Agent 准确率 < 45% |
| 6 | 评分校准 | `_check_calibration()` | 高分标的买入占比 < 60% |
| 7 | Agent 冗余 | `_check_agent_redundancy()` | 平均分差 < 8（限 top 5） |
| 8 | 辩论影响 | `_check_debate_impact()` | 实质性调整 < 30% |
| 9 | LLM 覆盖率 | `_check_llm_override_rate()` | LLM 覆盖 > 60% 或 < 10% |
| 10 | 仓位合理性 | `_check_position_sizing()` | 单只 > 25% 或总仓位 > 100% |

**累积数据：** `data/review/optimization.json`（保留最近 30 次运行）

---

## 四、完整执行流程（MainSchedulerAgent.run）

```
Phase 0:   宏观分析 (1次)
  MacroAnalystAgent.run()          → global_max_pos
Phase 0.5: 政策&流动性 (1次)
  MonetaryPolicyAgent.run()
  PolicyEventAgent.run()
Phase 0.75: 市场状态检测
  detect_market_state()            → "强趋势牛/震荡偏强/震荡偏弱/强趋势熊"
Phase 0.6: ETF分层
  _rank_etf_tiers()                → Tier 1/2/3（按成交量+波动率+动量排名）
─────────────────────────────────────────────
Phase 1:   独立研究 (每只 ETF 并行)
  └─ Tier 1 (全部 40 只): 12 Agent 并行 → 辩论 → 首席决策
  └─ Tier 2 (无，目前全部为 Tier 1)
  └─ Tier 3 (无，全部 LLM 分析)
─────────────────────────────────────────────
Phase 2.5: 组合约束
  总仓位归一化 (sum > 100% → 等比缩放)
Phase 2.6: 相关性约束
  _apply_correlation_constraint()  → 高相关(>0.8)ETF组总仓位≤30%
Phase 2.7: 增强回测
  _enhance_with_backtest()         → 逐ETF回测 + 组合汇总
Phase 2.8: 组合优化
  _portfolio_optimization_phase()  → 风险平价/均值-方差
─────────────────────────────────────────────
Phase 3:   报告输出
  ResearchReportGenerator.generate_full_report()
  → 控制台报告 + .xlsx 摘要 + .txt 完整报告
─────────────────────────────────────────────
Phase 4:   复盘
  ReviewManager.process()          → 快照 + T+1验证 + 准确率统计
─────────────────────────────────────────────
Phase 5:   个性化建议 + 自动调仓
  portfolio.advise()               → 基于实盘持仓的个性化建议
  autotrade.auto_trade()           → 模拟调仓指令
─────────────────────────────────────────────
Phase 6:   工作流总监审查
  WorkflowDirector.review()        → 10维度审查报告
```

---

## 五、数据流与关键文件

### 5.1 输入数据流

```
akshare (实时API)
  ├─ 日K线(ETF价格/成交量) → DataCollectAgent.get_etf_price()
  ├─ 指数估值(PE/PB百分位) → DataCollectAgent.get_index_val()
  ├─ 北向资金               → DataCollectAgent.get_north_flow()
  ├─ 涨停数据               → ak.stock_zt_pool_em()
  ├─ 解禁数据               → ak.stock_restricted_release_queue_sina()
  ├─ 市场成交额             → ak.stock_sse_deal_daily()
  ├─ 宏观数据(PMI/M2/LPR)   → ak.macro_china_pmi() 等
  └─ 全球信号(USDCNY/美债)  → ak.spot_quote() 等
```

### 5.2 缓存层

`PersistentCache` (SQLite, `data/cache.db`)：
- 各数据独立 TTL（ETF 价格 1h / 指数估值 2h / 市场成交 30min / 舆情 1h）
- 线程安全，WAL 模式
- 自动清理 7 天前的数据

### 5.3 输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `ETF_多智能体投研报告_{yyyymmdd}.txt` | 文本 | 每 Agent 分析原文 + 辩论记录 + 首席决策 |
| `ETF_多智能体投研报告_{yyyymmdd}.xlsx` | Excel | 摘要看板（操作/仓位/共识度） |
| `data/snapshots/{yyyymmdd}.json` | JSON | 每日快照（供 T+1 复盘） |
| `data/review/cumulative_stats.json` | JSON | 累计准确率统计 + Agent 权重 |
| `data/review/optimization.json` | JSON | WorkflowDirector 审查历史 |
| `data/portfolio.json` | JSON | 实盘持仓记录（手动更新） |

### 5.4 ETF 标的池 (`config.py`)

40 只 ETF，分三类：
- **宽基 (8只):** 上证50/沪深300/中证A500/中证500/中证1000/中证2000/创业板/科创50
- **行业 (18只):** 券商/银行/有色/煤炭/房地产/电力/基建/半导体/消费/医药 等
- **主题 (14只):** 军工/人工智能/大数据/机器人/传媒/农业/恒生科技/黄金/红利/纳指/中概互联 等

---

## 六、配置指南

### 6.1 核心开关 (`config.py`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_ENABLED` | `True` | LLM 多智能体模式总开关 |
| `DEBATE_ENABLED` | `True` | 辩论功能开关 |
| `MAIN_WORKERS` | `6` | 并行处理 ETF 数 |
| `AGENT_WORKERS` | `8` | 单 ETF 内 Agent 并发数 |

### 6.2 LLM 双模型配置 (`.env`)

```
# 深度模型（宏观/政策/首席决策/辩论）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 快速模型（价值/技术/情绪/资金/风控/行业/零售/跨市场/游资/解禁/形态/趋势）
QUICK_LLM_API_KEY=sk-xxx
QUICK_LLM_BASE_URL=https://api.deepseek.com/v1
QUICK_LLM_MODEL=deepseek-chat
```

### 6.3 辩论阈值

```python
DISAGREEMENT_SCORE_THRESHOLD = 18  # 评分差≥此值触发辩论
DISAGREEMENT_RATING_GAP = 2        # 评级级差≥此值触发辩论
```

### 6.4 Agent 记忆系统

```python
# 每个 Agent 基类(BaseLLMAgent)内置：
MEMORY_ENABLED = True   # 注入历史分析记录
MEMORY_DAYS = 20        # 检索近20天
```
- 宏观/货币/政策 Agent 跳过记忆（`MEMORY_ENABLED` 对全局/政策级无效）
- 记忆通过 `MemoryRetriever` 从 snapshot 检索

---

## 七、扩展指南

### 7.1 新增 Agent

1. 在 `agents/fundamental.py` 或 `agents/market.py` 中新建类
2. 继承 `BaseLLMAgent`，实现 `run()` 方法
3. 在 `agents/__init__.py` 中添加导出
4. 在 `scheduler.py` 的 `_research_single_etf()` 中添加 Agent 调用（Phase 1）
5. 在 `ChiefDecisionAgent.SCORING_AGENTS` 中添加名称（Phase 3 加权用）
6. **更新本文档**，在第三节中添加 Agent 条目

### 7.2 新增审查维度

1. 在 `WorkflowDirector` 中添加 `_check_xxx()` 方法
2. 在 `review()` 方法中添加调用
3. 在 `_generate_suggestions()` 中生成对应建议
4. 可选：在 `print_summary()` 中添加展示
5. **更新本文档**，在工作流总监表格中添加维度

### 7.3 修改 ETF 池

直接在 `config.py` 的 `ETF_POOL` 列表中增删改，格式：
```python
{"code": "510300", "name": "沪深300ETF", "type": "宽基", "index_code": "000300"}
```

### 7.4 切换 LLM 提供商

在 `.env` 中修改：
- `DEEPSEEK_API_KEY` → 改为 OpenAI/Claude/其他 API Key
- `DEEPSEEK_BASE_URL` → 改为对应 API 地址
- `DEEPSEEK_MODEL` → 改为模型名

---

## 八、运行方式

### 午后全量分析（15:00 后）
```bash
python etf-agent.py
```
全量分析 40 只 ETF，12 Agent + 辩论 + 首席决策 + 组合优化 + 自动调仓。

### 晨盘增量分析（8:15 开盘前）
```bash
python morning_update.py
```
轻量增量更新：加载昨日快照 → 宏观Agent更新 → 规则重评分 → 操盘指导。
**不调 LLM，纯规则评分**，适合开盘前快速获取操作要点。

### 流程对比

| | 晨盘 8:15 | 午后 15:30 |
|---|---|---|
| **入口** | `morning_update.py` / `run_morning.ps1` | `etf-agent.py` / `run_daily.ps1` |
| **分析范围** | 顶层Agent（宏观）增量 + 规则重评分 | 全量 12 Agent + 辩论 + 首席决策 |
| **LLM调用** | 仅宏观Agent | 全部 Agent + 辩论 |
| **输出** | 操盘指导文本（推送手机） | 完整投研报告 + 复盘 + 自动调仓 |
| **耗时** | ~2-5 分钟 | ~15-30 分钟 |

### 看板
```bash
streamlit run dashboard.py
```

### 自动调度
- **8:15** — `run_morning.ps1`（Windows Task Scheduler）
- **15:30** — `run_daily.ps1`（Windows Task Scheduler）+ GitHub Actions

---

## 九、版本记录

| 日期 | 变更 | 涉及文件 |
|------|------|----------|
| 2026-06-04 | 定时分析推送（晨盘8:15 + 午盘增强） | `morning_update.py`, `run_morning.ps1`, `decision.py`, `autotrade.py`, `run_daily.ps1` |
| 2026-06-03 | WorkflowDirector 深度审查 10 维度 | `director.py` |
| 2026-06-03 | WorkflowDirector 基础版本 | `director.py` |
| 2026-06-02 | PolicyEventAgent 接入实时政策新闻 | `agents/macro.py` |
| 2026-06-02 | Dashboard Plotly 交互图表 | `dashboard.py` |
| 2026-06-01 | Skills 驱动的投研管线升级 | 多文件 |
| 2026-06-01 | 辩论引擎升级为仲裁模式 | `debate.py` |
| 2026-05-31 | 自动化模拟交易引擎 | `autotrade.py` |
| 2026-05-31 | 实盘投资组合管理 | `portfolio.py` |

> **维护提醒：** 新增 Agent、新增审查维度、修改执行流程后，请同步更新本文档第三、四、七节。
