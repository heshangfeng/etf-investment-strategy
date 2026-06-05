# 晨盘 Delta 增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Make overnight market movements and news sentiment actually affect ETF scores in the morning update pipeline.

**Architecture:** Single-file change to `morning_update.py`. Three function rewrites (`_collect_overnight`, `_collect_morning_news`, `_map_impacts`) plus OVERNIGHT_MAP cleanup. No new files or dependencies.

**Tech Stack:** Python 3.14, akshare, FinBERT (already in `infra/sentiment_skill.py`), PersistentCache

---

### Task 1: Refactor OVERNIGHT_MAP — replace weight with delta_scale, remove dead code

**Files:**
- Modify: `morning_update.py:34-39`

- [ ] **Step 1: Update the OVERNIGHT_MAP constant**

Replace the old map (3 entries with unused `weight`, dead `hk_market`):

```python
OVERNIGHT_MAP = {
    "us_market": {"etfs": {"513100", "588000"}, "weight": 0.5},
    "a50":       {"etfs": {"510300", "510050", "159338"}, "weight": 0.4},
    "hk_market": {"etfs": {"513130", "513090"}, "weight": 0.5},
    "gold":      {"etfs": {"518880"}, "weight": 0.3},
}
```

With (4 entries, `delta_scale` replaces `weight`, `hk_market` removed):

```python
OVERNIGHT_MAP = {
    "us_market": {"etfs": {"513100", "588000"}, "delta_scale": 3.0},
    "a50":       {"etfs": {"510300", "510050", "159338"}, "delta_scale": 2.0},
    "gold":      {"etfs": {"518880"}, "delta_scale": 1.5},
}
```

**Rationale for delta_scale values:**
- `us_market` (3.0): Nasdaq 波动大（单日 1-3% 常见），3x 乘数使 delta 在 ±3~±9 分，合理影响评分
- `a50` (2.0): A50 波动相对小，2x 已足够
- `gold` (1.5): 黄金日波动通常 <1%，1.5x 让 delta 在 ±1.5 分以内，微调而非翻转评分

- [ ] **Step 2: Verify the change**

```bash
python -c "exec(open('morning_update.py').read()); print({k: v for k, v in OVERNIGHT_MAP.items()})"
```

Expected: No `weight` field, no `hk_market`, each entry has `delta_scale`.

---

### Task 2: Refactor `_collect_overnight()` — return prices AND change_pct

**Files:**
- Modify: `morning_update.py:72-115`

- [ ] **Step 1: Rewrite `_collect_overnight()` to collect change_pct alongside prices**

The function currently returns `{"us_market": {"djia": N, ...}, "a50": N, "commodities": {"gold": N, "oil": N}}`.

New return shape:
```python
{
    "us_market": {
        "djia": {"price": 51562, "change_pct": -0.3},
        "nasdaq": {"price": 741, "change_pct": -0.8},
        "sp500": {"price": 7564, "change_pct": -0.5},
    },
    "a50": {"price": 12800, "change_pct": 0.5},
    "commodities": {
        "gold": {"price": 2350, "change_pct": 0.2},
        "oil": {"price": 78, "change_pct": -1.2},
    },
}
```

Full rewrite:

```python
def _collect_overnight() -> dict:
    """采集隔夜外盘数据，返回价格及涨跌幅(%)"""
    from core.config import CACHE_ETF_PRICE  # reuse persistent cache for prev values

    result = {
        "us_market": {},
        "a50": None,
        "commodities": {},
    }
    try:
        import akshare as ak

        # ── 美股指数：用历史数据计算涨跌幅 ──
        for symbol, key in [(".DJI", "djia"), ("QQQ", "nasdaq")]:
            try:
                df = ak.index_us_stock_sina(symbol=symbol)
                if df is not None and len(df) >= 2:
                    closes = df["close"].values[-2:]
                    price = float(closes[-1])
                    prev = float(closes[-2])
                    change_pct = round((price - prev) / prev * 100, 2)
                    result["us_market"][key] = {"price": price, "change_pct": change_pct}
            except Exception:
                pass

        # ── 标普500期货 ──
        try:
            sp = ak.futures_global_spot_em()
            es = sp[sp["名称"].str.contains("标普", na=False)]
            if not es.empty:
                row = es.iloc[0]
                price = float(row["最新价"])
                chg = float(row.get("涨跌幅", 0) if "涨跌幅" in row.index else 0)
                result["us_market"]["sp500"] = {"price": price, "change_pct": chg}
        except Exception:
            pass

        # ── A50/黄金/原油：futures_foreign_commodity_realtime ──
        try:
            futures = ak.futures_foreign_commodity_realtime()
            # A50
            a50_row = futures[futures["symbol"].str.contains("A50", na=False)]
            if not a50_row.empty:
                price = float(a50_row["last_price"].iloc[0])
                # 通过缓存获取昨值
                cache = CACHE_ETF_PRICE  # reuse existing cache infra
                prev_key = "overnight_a50"
                prev = cache.get(prev_key)
                if prev is not None:
                    change_pct = round((price - prev) / prev * 100, 2)
                else:
                    change_pct = 0.0
                cache.set(prev_key, price)
                result["a50"] = {"price": price, "change_pct": change_pct}

            # 黄金/原油
            for _, row in futures.iterrows():
                s = str(row.get("symbol", ""))
                price = float(row.get("last_price", 0))
                if price == 0:
                    continue
                cache = CACHE_ETF_PRICE
                if "COMEX黄金" in s:
                    prev_key = "overnight_gold"
                    prev = cache.get(prev_key)
                    change_pct = round((price - prev) / prev * 100, 2) if prev is not None else 0.0
                    cache.set(prev_key, price)
                    result["commodities"]["gold"] = {"price": price, "change_pct": change_pct}
                elif "WTI原油" in s or "布伦特原油" in s:
                    prev_key = "overnight_oil"
                    prev = cache.get(prev_key)
                    change_pct = round((price - prev) / prev * 100, 2) if prev is not None else 0.0
                    cache.set(prev_key, price)
                    result["commodities"]["oil"] = {"price": price, "change_pct": change_pct}
        except Exception:
            pass
    except Exception as e:
        logger.warning("隔夜外盘采集失败: %s", e)
    return result
```

- [ ] **Step 2: Update the caller in main() to handle the new data shape**

In `main()` (line 474-476), update the overnight display:

```python
print(f"\n🌙 采集隔夜外盘...")
overnight = _collect_overnight()
us_keys = list(overnight.get("us_market", {}).keys())
# Display with change_pct
_us_parts = []
for k in us_keys:
    v = overnight["us_market"][k]
    chg = v.get("change_pct", 0)
    sign = "+" if chg >= 0 else ""
    _us_parts.append(f"{k}: {v['price']}({sign}{chg}%)")
print(f"  ✅ 美股指数: {', '.join(_us_parts) if _us_parts else '无'}")
```

---

### Task 3: Add FinBERT sentiment to `_collect_morning_news()`

**Files:**
- Modify: `morning_update.py:122-156`

- [ ] **Step 1: Import FinBertSentiment at top of file**

Add after existing imports (~line 18):

```python
from infra.sentiment_skill import FinBertSentiment, analyze_text_sentiment
```

- [ ] **Step 2: Rewrite `_collect_morning_news()` to attach sentiment per item**

```python
def _collect_morning_news() -> list[dict]:
    """采集早间舆情，每条附带 FinBERT 情感分数"""
    items = []
    seen_titles = set()

    # 初始化 FinBERT（惰性，仅在有可能需要时）
    finbert = None
    try:
        finbert = FinBertSentiment()
    except Exception:
        logger.warning("FinBERT 初始化失败，新闻情感将使用默认中性值")

    def _score(text: str) -> float:
        """FinBERT 情感打分，失败返回 50（中性）"""
        if finbert is None:
            return 50.0
        try:
            result = finbert.analyze(text[:200])
            return float(result.get("score", 50))
        except Exception:
            return 50.0

    # 1. 华尔街见闻
    try:
        r = requests.get(
            "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=8",
            headers=HEADERS, timeout=8)
        data = r.json()
        for item in data.get("data", {}).get("items", []):
            title = item.get("title", "") or item.get("content_text", "")
            if title and title[:30] not in seen_titles:
                seen_titles.add(title[:30])
                text = (title[:80] + " " + item.get("content_text", "")[:120])
                sentiment = _score(text)
                items.append({
                    "source": "wallstreetcn",
                    "title": title[:80],
                    "summary": item.get("content_text", "")[:120],
                    "sentiment": sentiment,
                })
    except Exception:
        logger.warning("华尔街见闻采集失败")

    # 2. akshare 财新
    try:
        import akshare as ak
        df = ak.stock_news_main_cx()
        if df is not None and len(df) > 0:
            for _, row in df.head(10).iterrows():
                summary = str(row.get("summary", ""))
                if summary and summary[:30] not in seen_titles:
                    seen_titles.add(summary[:30])
                    sentiment = _score(summary[:200])
                    items.append({
                        "source": "caixin",
                        "title": summary[:80],
                        "summary": summary,
                        "sentiment": sentiment,
                    })
    except Exception:
        logger.warning("akshare财新新闻采集失败")

    return items
```

---

### Task 4: Rewrite `_map_impacts()` — compute actual delta from overnight + news sentiment

**Files:**
- Modify: `morning_update.py:163-192`

- [ ] **Step 1: Rewrite `_map_impacts()`**

```python
def _map_impacts(overnight: dict, news: list[dict]) -> dict[str, dict]:
    """将外盘涨跌幅 + 新闻情感映射为每只ETF的评分调整(delta)"""
    impacts = {}

    # ── 外盘影响 ──
    us = overnight.get("us_market", {})
    # 取美股三大指数的平均涨跌幅作为 us_market 信号
    us_changes = [v["change_pct"] for v in us.values() if isinstance(v, dict) and v.get("change_pct") is not None]
    if us_changes:
        avg_us_change = sum(us_changes) / len(us_changes)
        scale = OVERNIGHT_MAP["us_market"]["delta_scale"]
        delta = round(avg_us_change * scale, 1)
        reason_parts = []
        for k, v in us.items():
            if isinstance(v, dict) and v.get("change_pct") is not None:
                sign = "+" if v["change_pct"] >= 0 else ""
                reason_parts.append(f"{k.upper()}{sign}{v['change_pct']}%")
        reason = " | ".join(reason_parts)
        for code in OVERNIGHT_MAP["us_market"]["etfs"]:
            impacts[code] = {"source": "overnight", "reason": f"隔夜美股: {reason}", "delta": delta}

    # A50 影响
    a50 = overnight.get("a50")
    if a50 and isinstance(a50, dict) and a50.get("change_pct") is not None:
        scale = OVERNIGHT_MAP["a50"]["delta_scale"]
        delta = round(a50["change_pct"] * scale, 1)
        sign = "+" if a50["change_pct"] >= 0 else ""
        reason = f"A50{sign}{a50['change_pct']}%"
        for code in OVERNIGHT_MAP["a50"]["etfs"]:
            if code in impacts:
                impacts[code]["delta"] = round(impacts[code]["delta"] + delta, 1)
                impacts[code]["reason"] += f" | {reason}"
            else:
                impacts[code] = {"source": "overnight", "reason": reason, "delta": delta}

    # 黄金影响
    gold = overnight.get("commodities", {}).get("gold")
    if gold and isinstance(gold, dict) and gold.get("change_pct") is not None:
        scale = OVERNIGHT_MAP["gold"]["delta_scale"]
        delta = round(gold["change_pct"] * scale, 1)
        sign = "+" if gold["change_pct"] >= 0 else ""
        reason = f"黄金{sign}{gold['change_pct']}%"
        for code in OVERNIGHT_MAP["gold"]["etfs"]:
            if code in impacts:
                impacts[code]["delta"] = round(impacts[code]["delta"] + delta, 1)
                impacts[code]["reason"] += f" | {reason}"
            else:
                impacts[code] = {"source": "overnight", "reason": reason, "delta": delta}

    # ── 新闻影响（关键词匹配 + FinBERT 情感）──
    # 逐条新闻匹配关键词，聚合每只ETF被命中的情感分数
    news_sentiments: dict[str, list[float]] = {}  # code -> [sentiment_scores]
    for n in news:
        text = n["title"] + " " + n.get("summary", "")
        sentiment = n.get("sentiment", 50.0)
        for keywords, etf_codes in NEWS_KEYWORD_MAP:
            for kw in keywords:
                if kw in text:
                    for code in etf_codes:
                        news_sentiments.setdefault(code, []).append(sentiment)
                    break

    for code, scores in news_sentiments.items():
        avg_sentiment = sum(scores) / len(scores)
        # sentiment 0-100, 中性在 50, delta = (sentiment - 50) / 10
        delta = round((avg_sentiment - 50) / 10, 1)
        kw_hit = next((kw for kws, _ in NEWS_KEYWORD_MAP for kw in kws
                      if kw in " ".join(n["title"] + n.get("summary", "") for n in news)), "")
        if code in impacts:
            impacts[code]["delta"] = round(impacts[code]["delta"] + delta, 1)
            impacts[code]["reason"] += f" | 新闻情感:{kw_hit}({avg_sentiment:.0f}分)"
        else:
            impacts[code] = {"source": "news", "reason": f"新闻:{kw_hit}(情感{avg_sentiment:.0f}分)", "delta": delta}

    return impacts
```

- [ ] **Step 2: Verify imports are complete**

The new code uses `OVERNIGHT_MAP` and `NEWS_KEYWORD_MAP` which are already defined above the function. No new imports needed.

---

### Task 5: Update `format_morning_guidance()` to show delta values

**Files:**
- Modify: `morning_update.py:396-402`

- [ ] **Step 1: Update the "受影响的ETF" section to show delta direction**

```python
    if impacted:
        lines.append("🔄 受影响的ETF")
        for r in impacted[:8]:
            imp = r.get("impact", {})
            d = imp.get("delta", 0)
            direction = "📈" if d > 0 else ("📉" if d < 0 else "➡️")
            lines.append(f"  {r['name']}: {imp.get('reason', '')} {direction} ({d:+.1f})")
        lines.append("")
```

---

### Task 6: Integration test — run morning_update.py

- [ ] **Step 1: Run the script and verify output**

```bash
$env:PYTHONIOENCODING='utf-8'; python morning_update.py
```

Expected:
- Overnight data now shows change_pct: `djia: 51562(+0.3%), nasdaq: 741(-0.8%), sp500: 7564(-0.5%)`
- News items carry sentiment scores
- "受影响ETF" shows delta values: `纳指ETF: 隔夜美股: NASDAQ-0.8% 📉 (-2.4)`
- "操作信号变化" shows real score changes driven by delta
- No errors or warnings (pre-existing warnings about .env are OK)

- [ ] **Step 2: Check diagnostics**

```bash
python -c "import py_compile; py_compile.compile('morning_update.py', doraise=True)"
```

Expected: No syntax errors.
