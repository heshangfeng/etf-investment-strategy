"""
ETF 智能投资分析系统 - 晨盘增量更新
8:15 AM 运行：隔夜外盘 + 早间舆情 → 影响映射 → 增量评分 → 持仓建议
不调LLM，纯规则评分，轻量快速。
"""
import json
import glob
import os
import numpy as np
from datetime import datetime

import requests
from core.config import SNAPSHOT_DIR, ETF_POOL, HEADERS
from core.data import DataCollectAgent
from agents import MacroAnalystAgent
from core.decision import ChiefDecisionAgent
from core.scheduler import MainSchedulerAgent

from infra.logger import get_logger; logger = get_logger(__name__)
from infra.sentiment_skill import FinBertSentiment

# 自动更新项目目录结构文档（静默，失败不影响分析）
try:
    from scripts.generate_structure import generate_markdown, write_structure_file
    _md = generate_markdown()
    write_structure_file(_md)
except Exception:
    logger.warning("自动更新项目目录结构文档失败", exc_info=True)
    pass


SNAPSHOT_PATH = SNAPSHOT_DIR

# ── 外盘 → ETF 映射 ──
OVERNIGHT_MAP = {
    "us_market": {"etfs": {"513100", "588000"}, "delta_scale": 3.0},
    "a50":       {"etfs": {"510300", "510050", "159338"}, "delta_scale": 2.0},
    "gold":      {"etfs": {"518880"}, "delta_scale": 1.5},
}

# 新闻关键词 → ETF 映射
NEWS_KEYWORD_MAP = [
    (["新能源", "光伏", "风电", "碳中和"], {"159875", "159806", "515790"}),
    (["半导体", "芯片", "光刻"], {"512480", "512760"}),
    (["医药", "医疗", "集采", "创新药"], {"159858", "512290"}),
    (["银行", "降准", "利率", "信贷"], {"512800"}),
    (["消费", "白酒", "食品"], {"512690", "159928"}),
    (["地产", "房地产", "楼市"], {"512200"}),
    (["券商", "证券", "资本市场"], {"512000"}),
    (["黄金", "避险"], {"518880"}),
    (["中概", "恒生", "港股"], {"513050", "513130", "513090"}),
    (["汽车", "新能源车"], {"516150", "159875"}),
    (["军工", "国防"], {"512660"}),
    (["AI", "人工智能", "大模型", "机器人"], {"159819", "516950"}),
    (["煤炭", "能源"], {"515220"}),
    (["有色", "稀土", "锂"], {"516670", "512400"}),
    (["农业", "粮食", "猪肉"], {"159825"}),
    (["传媒", "游戏", "短剧"], {"512980"}),
    (["电力", "电网", "绿电"], {"159611"}),
    (["基建", "水利", "交通"], {"515080"}),
    (["红利", "高股息", "分红"], {"510880"}),
    (["数字经济", "数据", "算力"], {"515070"}),
    (["元宇宙", "VR", "AR"], {"516220"}),
    (["创业板", "成长"], {"159915"}),
]


# ═══════════════════════════════════════════════
# ① 隔夜外盘数据采集
# ═══════════════════════════════════════════════

def _collect_overnight() -> dict:
    """采集隔夜外盘数据，返回价格及涨跌幅(%)"""
    from infra.cache import PersistentCache
    OVERNIGHT_CACHE = PersistentCache("overnight", default_ttl=86400)

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
                chg = float(row.get("涨跌幅", 0)) if "涨跌幅" in row.index else 0.0
                result["us_market"]["sp500"] = {"price": price, "change_pct": chg}
        except Exception:
            pass

        # ── A50/黄金/原油：futures_foreign_commodity_realtime ──
        futures = ak.futures_foreign_commodity_realtime()

        # A50 (standalone try/except)
        try:
            a50_row = futures[futures["symbol"].str.contains("A50", na=False)]
            if not a50_row.empty:
                price = float(a50_row["last_price"].iloc[0])
                prev_key = "overnight_a50"
                prev = OVERNIGHT_CACHE.get(prev_key)
                change_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
                OVERNIGHT_CACHE.put(prev_key, price)
                result["a50"] = {"price": price, "change_pct": change_pct}
        except Exception:
            pass

        # 黄金/原油 (standalone try/except)
        try:
            for _, row in futures.iterrows():
                s = str(row.get("symbol", ""))
                p = float(row.get("last_price", 0))
                if p == 0:
                    continue
                if "COMEX黄金" in s:
                    prev_key = "overnight_gold"
                    prev = OVERNIGHT_CACHE.get(prev_key)
                    change_pct = round((p - prev) / prev * 100, 2) if prev else 0.0
                    OVERNIGHT_CACHE.put(prev_key, p)
                    result["commodities"]["gold"] = {"price": p, "change_pct": change_pct}
                elif "WTI原油" in s or "布伦特原油" in s:
                    prev_key = "overnight_oil"
                    prev = OVERNIGHT_CACHE.get(prev_key)
                    change_pct = round((p - prev) / prev * 100, 2) if prev else 0.0
                    OVERNIGHT_CACHE.put(prev_key, p)
                    result["commodities"]["oil"] = {"price": p, "change_pct": change_pct}
        except Exception:
            pass
    except Exception as e:
        logger.warning("隔夜外盘采集失败: %s", e)
    return result


# ═══════════════════════════════════════════════
# ② 早间国内舆情
# ═══════════════════════════════════════════════

def _collect_morning_news() -> list[dict]:
    """采集早间舆情，每条附带 FinBERT 情感分数"""
    items = []
    seen_titles = set()

    def _score(text: str) -> float:
        """FinBERT 情感打分，失败返回 50（中性）"""
        try:
            score, _ = FinBertSentiment.analyze(text[:200])
            return float(score)
        except Exception:
            return 50.0

    # 1. 华尔街见闻实时快讯（API）
    try:
        r = requests.get(
            "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=8",
            headers=HEADERS, timeout=8)
        data = r.json()
        for item in data.get("data", {}).get("items", []):
            title = item.get("title", "") or item.get("content_text", "")
            if title and title[:30] not in seen_titles:
                seen_titles.add(title[:30])
                text = title[:80] + " " + item.get("content_text", "")[:120]
                sentiment = _score(text)
                items.append({
                    "source": "wallstreetcn",
                    "title": title[:80],
                    "summary": item.get("content_text", "")[:120],
                    "sentiment": sentiment,
                })
    except Exception:
        logger.warning("华尔街见闻采集失败")

    # 2. akshare 财新新闻
    try:
        import akshare as ak
        df = ak.stock_news_main_cx()
        if df is not None and len(df) > 0:
            for _, r in df.head(10).iterrows():
                summary = str(r.get("summary", ""))
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


# ═══════════════════════════════════════════════
# ③ 影响映射 — 将外盘涨跌幅 + 新闻情感映射为 delta
# ═══════════════════════════════════════════════

def _map_impacts(overnight: dict, news: list[dict]) -> dict[str, dict]:
    """将外盘涨跌幅 + 新闻情感映射为每只ETF的评分调整(delta)"""
    impacts = {}

    # ── 外盘影响 ──
    us = overnight.get("us_market", {})
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
    news_data: dict[str, list[tuple[float, str]]] = {}
    for n in news:
        text = n.get("title", "") + " " + n.get("summary", "")
        raw_s = n.get("sentiment")
        sentiment = raw_s if isinstance(raw_s, (int, float)) else 50.0
        for keywords, etf_codes in NEWS_KEYWORD_MAP:
            for kw in keywords:
                if kw in text:
                    for code in etf_codes:
                        news_data.setdefault(code, []).append((sentiment, kw))
                    break

    for code, items in news_data.items():
        scores = [s for s, _ in items]
        kw_hit = items[0][1]
        avg_sentiment = sum(scores) / len(scores)
        delta = round((avg_sentiment - 50) / 10, 1)
        if code in impacts:
            impacts[code]["delta"] = round(impacts[code]["delta"] + delta, 1)
            tag = f"新闻:{kw_hit}(情感{avg_sentiment:.0f}分)"
            impacts[code]["reason"] += f" | {tag}"
        else:
            impacts[code] = {"source": "news", "reason": f"新闻:{kw_hit}(情感{avg_sentiment:.0f}分)", "delta": delta}

    return impacts


# ═══════════════════════════════════════════════
# ④ 增量评分引擎
# ═══════════════════════════════════════════════

def re_score_etfs(snapshot_etfs: list, top: dict, impacts: dict) -> list[dict]:
    """用增量影响重算评分"""
    results = []
    for etf in snapshot_etfs:
        code = etf["code"]
        raw_agents = etf.get("agents", [])
        if not raw_agents:
            continue

        agent_scores = [
            {"name": a["n"], "score": a["sc"], "rating": a["rt"]}
            for a in raw_agents
        ]

        # 如果有舆情/外盘影响，注入一个虚拟Agent评分
        imp = impacts.get(code)
        if imp:
            base_score = sum(a["sc"] for a in raw_agents) / len(raw_agents)
            injected_score = float(np.clip(base_score + imp["delta"], 0, 100))
            agent_scores.append({
                "name": "晨间舆情信号",
                "score": injected_score,
                "rating": "看多" if injected_score >= 55 else "看空" if injected_score <= 45 else "中性",
            })

        result = ChiefDecisionAgent.compute_rule_score(
            agent_scores=agent_scores,
            etf_code=code,
            etf_name=etf["name"],
            etf_type=etf.get("type", ""),
            global_max_pos=top["global_max_pos"],
            market_state=top["market_state"],
        )

        # compute_rule_score 的 pos_map 对"持有""减持"返回0，这里补兜底
        pos_pct = result["position_pct"]
        if pos_pct == 0 and result["operation"] in ("持有", "长期持有"):
            pos_pct = 0.15
        elif pos_pct == 0 and result["operation"] in ("减持",):
            pos_pct = 0.05

        old_op = etf.get("operation", "")
        old_score = etf.get("final_score", 0)
        results.append({
            "code": code, "name": etf["name"], "type": etf.get("type", ""),
            "old_operation": old_op, "new_operation": result["operation"],
            "old_score": old_score, "new_score": result["final_score"],
            "score_delta": round(result["final_score"] - old_score, 1),
            "position_pct": pos_pct,
            "consensus": result["consensus"],
            "changed": old_op != result["operation"],
            "impact": imp,
        })

    return results


def apply_portfolio_constraints(results: list[dict]) -> list[dict]:
    """应用总仓位归一化和相关性约束"""
    total_pos = sum(r["position_pct"] for r in results)
    if total_pos > 1.0:
        scale = 1.0 / total_pos
        for r in results:
            r["position_pct"] = round(r["position_pct"] * scale, 4)

    actionable = [r for r in results if r["position_pct"] > 0]
    if len(actionable) >= 2:
        price_data = {}
        for r in actionable:
            try:
                df = DataCollectAgent.get_etf_price(r["code"])
                if len(df) >= 60:
                    price_data[r["code"]] = df["close"].pct_change().dropna().tail(60).values
            except Exception:
                logger.warning("获取价格数据计算相关性失败", exc_info=True)
                pass

        if len(price_data) >= 2:
            codes = list(price_data.keys())
            price_matrix = np.array([price_data[c] for c in codes])
            corr_matrix = np.corrcoef(price_matrix)
            n = len(codes)
            visited = set()
            for i in range(n):
                if i in visited:
                    continue
                group = [i]
                visited.add(i)
                for j in range(i + 1, n):
                    if j not in visited and corr_matrix[i][j] > 0.8:
                        group.append(j)
                        visited.add(j)
                if len(group) > 1:
                    group_codes = [codes[idx] for idx in group]
                    group_pos = sum(r["position_pct"] for r in results if r["code"] in group_codes)
                    if group_pos > 0.3:
                        scale = 0.3 / group_pos
                        for r in results:
                            if r["code"] in group_codes:
                                r["position_pct"] = round(r["position_pct"] * scale, 4)

    return results


# ═══════════════════════════════════════════════
# ⑤ 持仓操作建议
# ═══════════════════════════════════════════════

def _portfolio_advice(results: list[dict]) -> list[dict]:
    """根据实际持仓+新信号生成个性化操作建议"""
    advices = []
    try:
        from trading.portfolio import load, _price
        pf = load()
        result_map = {r["code"]: r for r in results}
        for h in pf.holdings:
            r = result_map.get(h.code)
            if not r:
                continue
            cp = _price(h.code, h.avg_cost)
            pnl_pct = (cp - h.avg_cost) / h.avg_cost * 100
            old_op = r["old_operation"]
            new_op = r["new_operation"]
            changed = r["changed"]
            impact = r.get("impact")

            if changed:
                suggestion = f"操作信号{old_op}→{new_op}，请关注"
            elif new_op in ("减持", "卖出", "强烈卖出"):
                suggestion = "信号偏空，冲高可考虑减仓"
            elif new_op in ("买入", "强烈买入"):
                suggestion = "信号偏多，可关注建仓机会"
            else:
                suggestion = "维持现状"

            if impact and impact.get("reason"):
                suggestion += f"（受{impact['reason']}影响）"

            advices.append({
                "code": h.code, "name": h.name,
                "shares": h.shares, "avg_cost": h.avg_cost,
                "price": cp, "pnl_pct": round(pnl_pct, 2),
                "old_operation": old_op, "new_operation": new_op,
                "changed": changed, "suggestion": suggestion,
            })
    except Exception as e:
        logger.warning("持仓建议生成失败: %s", e)
    return advices


# ═══════════════════════════════════════════════
# ⑥ 推送格式
# ═══════════════════════════════════════════════

def format_morning_guidance(overnight: dict, news: list[dict], results: list[dict],
                             advices: list[dict], top: dict, snap_date: str) -> str:
    """组装晨盘操盘指导推送文本（手机优化版）"""
    lines = [f"🏆 ETF晨盘操盘指导 {datetime.now().strftime('%m/%d')}\n"]

    # 🌙 隔夜外盘 — 一行搞定
    us = overnight.get("us_market", {})
    has_overnight = any(isinstance(us.get(k), dict) for k in ("djia", "nasdaq", "sp500"))
    parts = []
    label_map = {"djia": "道指", "nasdaq": "纳指", "sp500": "标普"}
    for k, label in label_map.items():
        v = us.get(k)
        if isinstance(v, dict) and v.get("price"):
            chg = v.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(f"{label}{v['price']:.0f}({sign}{chg:.1f}%)")
    if parts:
        lines.append("🌙 " + " ".join(parts))

    # 📰 早间要闻 — 精简到 4 条
    if news:
        shown = set()
        items = []
        for n in news:
            key = n.get("title", "")[:15]
            if key and key not in shown:
                shown.add(key)
                items.append(n.get("title", "")[:50])
        if items:
            for item in items[:4]:
                lines.append(f"📰 {item}")

    # 🔄 受影响的ETF — 紧凑显示
    impacted = [r for r in results if r.get("impact") and r.get("impact", {}).get("reason")]
    if impacted:
        for r in impacted[:6]:
            imp = r.get("impact", {})
            d = imp.get("delta", 0)
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
            lines.append(f"🔄 {r['name']}{arrow}{d:+.1f}")

    # 📊 操作信号变化 — 只显示变化的
    changed = [r for r in results if r["changed"]]
    if changed:
        for r in changed[:5]:
            lines.append(f"📊 {r['name']}: {r['old_operation']}→{r['new_operation']}")

    # 👤 持仓建议 — 每只一行
    if advices:
        for a in advices:
            flag = "⚠️" if a["changed"] else ""
            lines.append(f"👤 {a['name']}")
            lines.append(f"   信号 {a['old_operation']}→{a['new_operation']}{flag}")
            lines.append(f"   盈亏 {a['pnl_pct']:+.1f}% | {a['price']:.4f}")
            lines.append(f"   建议 {a['suggestion']}")
        lines.append("")

    lines.append(f"📊 {snap_date}快照+今晨舆情")
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 启动函数
# ═══════════════════════════════════════════════

def run_top_level_agents() -> dict:
    """运行宏观Agent + 检测市场状态"""
    macro_agent = MacroAnalystAgent()
    macro_report = macro_agent.run()
    global_max_pos = macro_report.score / 100
    market_state = MainSchedulerAgent().detect_market_state()
    macro_summary = (macro_report.analysis[:120] if macro_report.analysis else "")
    return {
        "macro_score": macro_report.score,
        "global_max_pos": global_max_pos,
        "market_state": market_state,
        "macro_summary": macro_summary,
    }


def load_latest_snapshot() -> dict | None:
    """加载最新的历史快照"""
    files = sorted(glob.glob(os.path.join(SNAPSHOT_PATH, "*.json")))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def get_snapshot_date(data: dict) -> str:
    return data.get("date", data.get("snapshot_date", "unknown"))


def main():
    snapshot = load_latest_snapshot()
    if snapshot is None:
        print("尚无历史快照数据，请先运行午后全量分析 (python etf-agent.py)")
        return

    snap_date = get_snapshot_date(snapshot)
    print(f"📂 加载快照: {snap_date}, {len(snapshot.get('etfs', []))} 只ETF")

    # ① 隔夜外盘
    print(f"\n🌙 采集隔夜外盘...")
    overnight = _collect_overnight()
    us = overnight.get("us_market", {})
    us_keys = list(us.keys())
    _us_parts = []
    for k in us_keys:
        v = us[k]
        chg = v.get("change_pct", 0)
        sign = "+" if chg >= 0 else ""
        _us_parts.append(f"{k}: {v['price']:.0f}({sign}{chg:.1f}%)")
    print(f"  ✅ 美股指数: {', '.join(_us_parts) if _us_parts else '无'}")

    # ② 早间舆情
    print(f"\n📰 采集早间舆情...")
    news = _collect_morning_news()
    print(f"  ✅ 新闻: {len(news)} 条")

    # ③ 影响映射
    print(f"\n🔄 映射影响...")
    impacts = _map_impacts(overnight, news)
    print(f"  ✅ 受影响ETF: {len(impacts)} 只")

    # ④ 增量评分
    print(f"\n📊 增量重评分...")
    top = run_top_level_agents()
    etfs = snapshot.get("etfs", [])
    results = re_score_etfs(etfs, top, impacts)
    results = apply_portfolio_constraints(results)
    changed_count = sum(1 for r in results if r["changed"])
    print(f"  ✅ 操作变化: {changed_count} 只")

    # ⑤ 持仓建议
    print(f"\n👤 生成持仓建议...")
    advices = _portfolio_advice(results)
    print(f"  ✅ 持仓: {len(advices)} 只")

    # ⑥ 输出推送
    guidance = format_morning_guidance(overnight, news, results, advices, top, snap_date)
    print("\n" + "=" * 60)
    print(guidance)
    print("=" * 60)


if __name__ == "__main__":
    main()
