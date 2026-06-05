# 晨盘增量分析：Delta 增强方案

**日期**: 2026-06-05
**状态**: 已批准
**改动范围**: `morning_update.py` 单文件

---

## 问题

晨盘增量分析中，`_map_impacts()` 返回的 `delta` 始终为 0：

- 隔夜外盘数据只采集不计算涨跌幅，`delta` 硬编码为 0
- 新闻只做关键词匹配，不做情感方向判断，`delta` 硬编码为 0
- `re_score_etfs()` 注入的"晨间舆情信号"虚拟 Agent 评分 = 已有 Agent 评分均值，实际无新增信息
- `OVERNIGHT_MAP` 中的 `weight` 字段从未使用，`hk_market` 映射无数据源

唯一真正影响评分变化的是 `_time_series_momentum_score()`（权重 0.35，最多影响 ±3.5 分）。

---

## 改动

### 1. `_collect_overnight()` — 返回涨跌幅

修改函数签名/返回结构，使每个市场数据同时包含 `price` 和 `change_pct`。

| 数据源 | 获取方式 |
|--------|---------|
| 美股指数 (`index_us_stock_sina`) | 取 DataFrame 倒数第一行 `close`（当日）和倒数第二行 `close`（前一日），计算 `(today - yesterday) / yesterday * 100` |
| 标普500期货 (`futures_global_spot_em`) | 已有"最新价"列，取"涨跌幅"列（百分比数值） |
| A50 期货 (`futures_foreign_commodity_realtime`) | 用 `PersistentCache("overnight_prev")` 存储昨值→今晨读取计算 `change_pct` |
| 黄金/原油 (`futures_foreign_commodity_realtime`) | 同上 |

**容错**: 任一数据源获取涨跌幅失败时，该市场 `change_pct = 0`，不影响其他市场。

### 2. `OVERNIGHT_MAP` — 替换 weight 为 delta_scale

删除未使用的 `weight` 字段和 `hk_market` 条目，新增 `delta_scale`：

```python
OVERNIGHT_MAP = {
    "us_market": {"etfs": {"513100", "588000"}, "delta_scale": 3.0},
    "a50":       {"etfs": {"510300", "510050", "159338"}, "delta_scale": 2.0},
    "gold":      {"etfs": {"518880"}, "delta_scale": 1.5},
}
```

`delta_scale` 含义：外盘每变动 1 个百分点，ETF 评分变动 N 分。

### 3. `_collect_morning_news()` — 产出带情感分数的新闻

`_collect_morning_news()` 返回每条新闻时追加 `sentiment` 字段：

```python
{"source": "wallstreetcn", "title": "...", "summary": "...", "sentiment": 65.0}
# sentiment: 0-100, <40=偏空 >60=偏多, 通过 FinBERT.analyze(title + summary) 获取
```

FinBERT 在采集循环中逐条调用，失败时该条 `sentiment = 50`（中性）。

### 4. `_map_impacts()` — 计算实际 delta

**外盘部分** — 用 `change_pct × delta_scale`：

```python
if "change_pct" in us_market_data:
    delta = round(us_market_data["change_pct"] * OVERNIGHT_MAP["us_market"]["delta_scale"], 1)
    for code in OVERNIGHT_MAP["us_market"]["etfs"]:
        impacts[code] = {"source": "overnight", "reason": f"纳指{us_market_data['change_pct']:+.1f}%", "delta": delta}
```

**新闻部分** — 逐条用 FinBERT 情感分数 → 按 ETF 聚合平均：

```python
# 对每条匹配关键词的新闻，取其 sentiment 分数
# 同一 ETF 被多条新闻命中时，取各条 sentiment 均值
# delta = round((avg_sentiment - 50) / 10, 1)  → 范围约 -5 ~ +5
```

**容错**: FinBERT 加载失败时所有新闻 `sentiment = 50`，`delta = 0`（退化到当前行为），不影响外盘 delta。

### 5. 缓存昨值（用于 A50/黄金/原油涨跌幅）

新增 `PersistentCache("overnight_prev")`：

- 每次 `_collect_overnight()` 运行时，把当天采集到的 A50/黄金/原油价格写入缓存，key 为 `"{品种}_20260605"`
- 下次运行时读取前一天记录对比算 `change_pct`
- 首日无缓存时 `change_pct = 0`

### 4. 清理

- 删除 `OVERNIGHT_MAP` 的 `weight` 字段（死代码）
- 删除 `hk_market` 条目（无数据源）

---

## 影响分析

### 正向影响

- 纳指隔夜跌 2% → 纳指ETF delta = -6.0，评分从 55 → 49，操作从"持有"变"减持"
- A50 涨 1% → 沪深300 delta = +2.0，评分从 48 → 50，操作从"减持"变"持有"
- 正面新闻命中 → FinBERT 分数 75 → delta = +2.5 → 评分微调向上
- 负面新闻命中 → FinBERT 分数 20 → delta = -3.0 → 评分下调

### 风险

- `futures_foreign_commodity_realtime` 的 API 字段可能变化 → 已有 try/except 兜底
- FinBERT 加载慢（约 3-5 秒）+ 推理 18 条新闻约 2-3 秒 → 总耗时增加约 5-8 秒，可接受
- FinBERT 模型文件可能缺失 → try/except 兜底，退化到 `delta = 0`

### 非影响

- 不修改其他文件
- 不增加 Python 依赖
- 不改变 LLM 调用（纯规则不变）

---

## 验证

1. `_collect_overnight()` 返回数据中 `change_pct` 不为 None
2. `_map_impacts()` 返回的 impact 对象中 `delta != 0`
3. `re_score_etfs()` 计算后，受外盘影响的 ETF 评分有可见变化
4. `run_morning.ps1` 运行无报错
