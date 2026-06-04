# 定时分析推送系统 - 设计规格

## 1. 概述

在现有 `scheduler.py`（午后全量分析）之外，新增**早盘增量分析**（8:15 AM）推送，并增强**午后复盘推送**内容。两次推送均通过 PushDeer 推送至手机。

### 现有机制

| 时段 | 已有 | 缺什么 |
|------|------|--------|
| 8:15 AM | ❌ 无 | 晨盘操盘指导推送 |
| 15:30 PM | ✅ `run_daily.ps1` → `etf-agent.py` → PushDeer | 推送内容缺少模拟盘当日盈亏 |

### 新增文件

| 文件 | 用途 |
|------|------|
| `morning_update.py` | 晨盘增量分析入口 - 加载昨日快照 + 顶层Agent增量和重决策 |
| `run_morning.ps1` | 晨盘编排脚本 - 执行分析 + PushDeer推送 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `run_daily.ps1` | 推送内容增加模拟盘当日盈亏（从 performance.json 读取） |

---

## 2. 晨盘增量分析（8:15 AM）

### 2.1 设计原则

- **不复跑全量 ETF 分析**：40只ETF的12维Agent分析耗时约15-30分钟，晨盘时间窗口不够
- **只增量和重决策**：复用昨日 Agent 评分数据，只更新宏观/政策/跨市场背景和首席决策
- **输出紧凑**：手机推送适合简短、可操作的文本

### 2.2 数据处理流程

```
┌─────────────────────────────────────────────────────┐
│                 morning_update.py                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. 加载最新快照 ← data/snapshots/{latest}.json     │
│     ├─ ETF列表: code, name, type, agents[]评分,     │
│     │   final_score, position_pct, operation,        │
│     │   consensus, close_price                      │
│     └─ market_volume_bn (昨日市场成交额)             │
│                                                      │
│  2. 顶层Agent增量运行                                │
│     ├─ MacroAnalystAgent.run()       → macro_score  │
│     ├─ MonetaryPolicyAgent.run()     → mon_report   │
│     ├─ PolicyEventAgent.run()        → pol_report   │
│     ├─ CrossMarketAgent.run()        → xmkt_report  │
│     │   (隔夜外盘: 美股/港股/中概/A50/人民币汇率)    │
│     └─ detect_market_state()         → market_state │
│                                                      │
│  3. 晨盘规则评分 (纯数学, 不调LLM)                    │
│     ├─ 注意: 快照agents仅存 {n, rt, sc, src},        │
│     │   缺少analysis/key_factors等文本字段,           │
│     │   无法完整重建AgentReport → 不调LLM路径         │
│     │                                                │
│     ├─ 对每只ETF, 提取昨日agent scores               │
│     ├─ 应用 ChiefDecisionAgent 的规则评分逻辑:        │
│     │   ├─ RiskAgent反转 (100 - score)               │
│     │   ├─ z-score标准化 → 50+z*15                   │
│     │   ├─ 置信度校准 + 动态加权平均                 │
│     │   ├─ 时间序列动量得分 (调用get_etf_price)       │
│     │   ├─ 分歧折扣 (consensus调整)                  │
│     │   └─ 市场状态乘数 + 凯利仓位 → 新建议仓位      │
│     ├─ global_max_pos 由新macro_score决定            │
│     └─ 对比新旧操作, 标记"变化"和"不变"              │
│                                                      │
│  4. 组合约束 (同 scheduler.py Phase 2.5-2.6)        │
│     ├─ 总仓位归一化 (≤100%)                          │
│     └─ 相关性约束 (相关系数>0.8的ETF合计≤30%)        │
│                                                      │
│  5. 输出操盘指导文本                                 │
│     ├─ 市场状态 + 宏观要点  (1-2行)                  │
│     ├─ 操作信号最强TOP3  (强烈买入/买入)             │
│     ├─ 风险信号  (减持/卖出)                         │
│     ├─ 今日重点变化  (评分/操作 vs 昨日)             │
│     └─ 建议总仓位                                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

> **注意**: ChiefDecisionAgent 的规则评分逻辑可在 `morning_update.py` 中以内联方式实现（约60行），或提取为 `decision.py` 的一个静态方法供复用。选择后者可减少重复代码。

### 2.3 快照数据结构依赖

从 `data/snapshots/{yyyymmdd}.json` 加载的每条ETF记录：

```python
{
    "code": "510300",
    "name": "沪深300ETF",
    "type": "宽基",
    "operation": "买入",
    "holding_period": "中期(1-3月)",
    "final_score": 72.5,
    "final_rating": "看多",
    "position_pct": 0.25,
    "consensus": "基本一致",
    "close_price": 3.85,
    "agents": [
        {"n": "价值估值智能体", "rt": "中性", "sc": 50, "src": "llm"},
        {"n": "技术趋势智能体", "rt": "看多", "sc": 70, "src": "llm"},
        ...
    ]
}
```

### 2.4 推送内容格式

```
🏆 今日操盘指导 · 6月4日

【市场状态】震荡偏强 | 总仓位上限 75%
宏观: 国内流动性宽松延续, 海外美股小幅回调

🔥 强烈买入 (3)
  159915 创业板ETF  仓位25%
  588000 科创50ETF  仓位20%
  512100 中证1000ETF 仓位15%

📈 买入 (5)
  510300 沪深300ETF  仓位15%
  159338 中证A500ETF 仓位12%
  ...

⚠️ 风险信号
  510050 上证50ETF → 减持 (评分52, 共识分歧)
  563300 中证2000ETF → 卖出 (评分32)

📊 变化提示
  159915: 持有→强烈买入 (评分55→78, +宏观共振)
  510300: 买入→持有 (评分68→62, 技术走弱)

【建议总仓位】65%
```

### 2.5 编排脚本 `run_morning.ps1`

```powershell
# 触发器: Windows Task Scheduler 每日 8:15
# 程序: powershell
# 参数: -ExecutionPolicy Bypass -File run_morning.ps1

流程:
  1. python morning_update.py 2>&1  → 捕获输出
  2. 从输出提取操盘要点摘要 (~1000字内推送)
  3. PushDeer推送: "🏆 今日操盘指导\n\n{摘要}"
  4. 日志写入 logs/ 目录
```

---

## 3. 午后复盘推送增强（15:30 PM）

### 3.1 修改 `run_daily.ps1`

**当前推送内容**（保留不变）：
```
【策略表现】初始资金 / 当前总值 / 总收益率 / 最大回撤
【调仓明细】买入/卖出清单
```

**新增内容**（在【策略表现】下方追加）：
```
【模拟盘当日盈亏】
当日盈亏: +1,850 (+1.67%)
累计盈亏: +12,500
持仓市值: 78,500
现金余额: 34,000
```

### 3.2 数据来源

从 `data/performance.json` 读取：

```python
{
    "current_value": 112500,
    "cash": 34000,
    "market_value": 78500,
    "total_return_pct": 12.50,
    "max_drawdown_pct": -8.32,
    "daily_pnl": 1850,          # 当日盈亏 (新增)
    "daily_pnl_pct": 1.67,      # 当日收益率 (新增)
    "cumulative_pnl": 12500,    # 累计盈亏 (新增)
}
```

### 3.3 `autotrade.py` 修改

在 `performance.json` 写入时增加 `daily_pnl`、`daily_pnl_pct`、`cumulative_pnl` 字段，从 `trade_log.json` 最近两天的 snapshots 差值计算。

---

## 4. 调度配置

### Windows Task Scheduler — 晨盘 (新建)

| 项目 | 值 |
|------|-----|
| 名称 | ETF Morning Analysis |
| 触发器 | 每天, 8:15 |
| 操作 | 启动程序 |
| 程序 | `powershell` |
| 参数 | `-ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_morning.ps1"` |
| 起始于 | `D:\project\ETF投资策略` |

### Windows Task Scheduler — 午后 (已有, 不变)

| 项目 | 值 |
|------|-----|
| 名称 | ETF Daily Analysis |
| 触发器 | 每天, 15:30 |
| 操作 | 启动程序 |
| 程序 | `powershell` |
| 参数 | `-ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_daily.ps1"` |

---

## 5. 文件变更清单

### 新增

| 文件 | 估算行数 | 说明 |
|------|---------|------|
| `morning_update.py` | ~200 | 晨盘增量分析主逻辑 |
| `run_morning.ps1` | ~80 | 晨盘编排+推送脚本 |

### 修改

| 文件 | 改动量 | 说明 |
|------|-------|------|
| `run_daily.ps1` | ~15行 | 增加模拟盘当日盈亏推送段落 |
| `autotrade.py` | ~10行 | performance.json 增加 daily_pnl/daily_pnl_pct/cumulative_pnl 字段 |

---

## 6. 边界情况

| 场景 | 处理方式 |
|------|---------|
| 无历史快照（首次运行） | `morning_update.py` 提示"尚无历史数据，请先运行午盘全量分析"后退出 |
| 晨盘8:15时快照还是前天的 | 检查快照日期，非昨日数据时提示"数据可能陈旧"但继续执行 |
| PushDeer推送失败 | 记录日志，不阻塞流程 |
| 增量分析中顶层Agent LLM失败 | fallback 到规则评分（同 scheduler.py 的宏观看板模式） |
| 模拟盘无交易记录（新用户） | 推送中跳过模拟盘盈亏段落，不发空内容 |
