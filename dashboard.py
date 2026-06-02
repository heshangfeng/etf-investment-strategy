"""
ETF 智能投研看板 - Streamlit Dashboard

启动: streamlit run dashboard.py
"""
import json
import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── 页面配置 ──
st.set_page_config(page_title="ETF 智能投研看板", layout="wide")

from config import OUTPUT_DIR, SNAPSHOT_DIR, REVIEW_DIR
SNAPSHOT_DIR = Path(__file__).parent / SNAPSHOT_DIR
REVIEW_DIR = Path(__file__).parent / REVIEW_DIR
OUTPUT_DIR = Path(__file__).parent / OUTPUT_DIR
REPORT_PREFIX = "ETF_多智能体投研报告_"


# ── 辅助函数 ──

def load_snapshots() -> list[dict]:
    """加载所有历史快照，按日期排序。"""
    snaps = []
    for fpath in sorted(glob.glob(str(SNAPSHOT_DIR / "*.json"))):
        with open(fpath, "r", encoding="utf-8") as f:
            snaps.append(json.load(f))
    return snaps


def find_latest_report_date() -> str | None:
    """找到最新报告文件的日期。"""
    files = sorted(glob.glob(str(OUTPUT_DIR / f"{REPORT_PREFIX}*.xlsx")))
    if not files:
        return None
    name = os.path.basename(files[-1])
    return name.replace(REPORT_PREFIX, "").replace(".xlsx", "")[:8]  # 只取日期


def rating_color(rating: str) -> str:
    mapping = {
        "强烈看多": "#e74c3c",
        "看多": "#e67e22",
        "中性": "#f1c40f",
        "看空": "#27ae60",
        "强烈看空": "#2ecc71",
    }
    return mapping.get(rating, "#95a5a6")


def score_color(score: float) -> str:
    if score >= 70:
        return "red"
    elif score >= 55:
        return "orange"
    elif score >= 45:
        return "gray"
    elif score >= 30:
        return "green"
    return "green"


def render_rating_badge(rating: str) -> str:
    colors = {"强烈看多": "red", "看多": "orange", "中性": "gray",
              "看空": "green", "强烈看空": "green"}
    c = colors.get(rating, "gray")
    return f":{c}[{rating}]"


# ── 数据加载 ──

@st.cache_data(ttl=60)
def load_data():
    """加载快照数据并返回结构化 dict。"""
    snaps = load_snapshots()
    if not snaps:
        return None
    latest = snaps[-1]
    date_str = latest.get("date", "未知")
    etfs = latest.get("etfs", [])

    # 加载累积统计
    cum_stats = {}
    cum_path = REVIEW_DIR / "cumulative_stats.json"
    if cum_path.exists():
        with open(cum_path, "r", encoding="utf-8") as f:
            cum_stats = json.load(f)

    return {
        "date": date_str,
        "market_volume": latest.get("market_volume_bn", 0),
        "etfs": etfs,
        "snap_count": len(snaps),
        "cum_stats": cum_stats,
    }


# ── 主界面 ──

def main():
    data = load_data()
    if data is None:
        st.warning("暂无数据。请先运行 etf-agent.py 生成报告。")
        st.info(f"期望快照目录: {SNAPSHOT_DIR}")
        return

    etfs = data["etfs"]
    date_str = data["date"]
    st.title(f"📊 ETF 智能投研看板 · {date_str}")

    # ── 顶部指标行 ──
    avg_score = sum(e["final_score"] for e in etfs) / len(etfs) if etfs else 0
    total_pos = sum(e.get("position_pct", 0) for e in etfs)
    long_count = sum(1 for e in etfs if e.get("position_pct", 0) > 0)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ETF 总数", len(etfs))
    col2.metric("平均得分", f"{avg_score:.1f}")
    col3.metric("建议持仓", long_count)
    col4.metric("建议总仓位", f"{total_pos*100:.1f}%")
    col5.metric("全市场成交额", f"{data['market_volume']:.0f}亿")

    # ── 筛选 ──
    types = list(set(e.get("type", "未知") for e in etfs))
    selected_type = st.sidebar.selectbox("板块类型", ["全部"] + sorted(types))
    filtered = [e for e in etfs if selected_type == "全部" or e.get("type") == selected_type]

    # ── ETF 表格 ──
    st.subheader("ETF 持仓表")
    rows = []
    for e in filtered:
        rows.append({
            "代码": e["code"],
            "名称": e["name"],
            "类型": e.get("type", ""),
            "评级": e.get("final_rating", ""),
            "得分": e["final_score"],
            "仓位": f"{e.get('position_pct', 0)*100:.1f}%",
            "操作": e.get("operation", ""),
            "持有周期": e.get("holding_period", ""),
            "共识度": e.get("consensus", ""),
            "现价": e.get("close_price", ""),
        })
    df = pd.DataFrame(rows)
    df.index = df["代码"]  # 用代码做索引方便选中

    cell_hover = {"selector": "td:hover", "props": "background-color: #ffffcc;"}
    styled = df.style.map(
        lambda v: f"color: {rating_color(v)}; font-weight: bold;" if v in ("强烈看多", "看多", "中性", "看空", "强烈看空") else "",
        subset=["评级"]
    ).map(
        lambda v: f"color: {score_color(float(v.rstrip('%')))};" if isinstance(v, str) and v.endswith('%') else "",
        subset=["仓位"]
    )
    st.dataframe(styled, width="stretch", height=min(60 + len(df) * 35, 600))

    # ── 单个 ETF 详情 ──
    st.subheader("ETF 详情")
    selected = st.selectbox("选择标的查看Agent分析", [""] + [f"{e['code']} {e['name']}" for e in etfs])
    if selected:
        code = selected.split()[0]
        etf = next((e for e in etfs if e["code"] == code), None)
        if etf:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**{etf['name']} ({etf['code']})** — {etf.get('type', '')}")
                st.markdown(f"最终评级: {render_rating_badge(etf.get('final_rating', ''))}  |  "
                            f"得分: **{etf['final_score']}**  |  "
                            f"建议仓位: {etf.get('position_pct', 0)*100:.1f}%")
                st.markdown(f"操作: {etf.get('operation', '')}  |  "
                            f"持有周期: {etf.get('holding_period', '')}  |  "
                            f"共识度: {etf.get('consensus', '')}")
            with col_b:
                st.metric("最新价", etf.get("close_price", "N/A"))

            # Agent 评分明细
            agents = etf.get("agents", [])
            if agents:
                st.markdown("**Agent 评分明细**")
                agent_rows = []
                for a in agents:
                    agent_rows.append({
                        "智能体": a.get("n", ""),
                        "评级": a.get("rt", ""),
                        "得分": a.get("sc", 0),
                        "来源": "LLM" if a.get("src") == "llm" else "规则",
                    })
                df_a = pd.DataFrame(agent_rows)
                st.dataframe(df_a.style.map(
                    lambda v: f"color: {rating_color(v)}; font-weight: bold;" if v in ("强烈看多", "看多", "中性", "看空", "强烈看空") else "",
                    subset=["评级"]
                ), width="stretch", hide_index=True)

    # ── 底部统计 ──
    st.divider()
    col_x, col_y = st.columns(2)
    with col_x:
        st.caption(f"历史快照: {data['snap_count']} 天")
    with col_y:
        cum = data.get("cum_stats", {})
        if cum:
            st.caption(f"累计复盘: {cum.get('total_reviews', 0)} 次 | "
                       f"准确率: {cum.get('overall_accuracy', 0):.1f}%")

    # ── 投资组合面板（实际持仓） ──
    st.divider()
    st.subheader("我的投资组合（实际持仓）")
    st.caption("顶部"建议总仓位"=系统推荐 ｜ 此处"实际仓位"=你的真实持仓比例，自动交易后两者应基本一致")
    try:
        from portfolio import load, _price
        pf = load()
        col_a, col_b, col_c = st.columns(3)
        cash = pf.cash
        market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
        total = cash + market_value
        col_a.metric("总资产", f"{total:.0f}")
        col_b.metric("持仓市值", f"{market_value:.0f}")
        col_c.metric("实际仓位", f"{(1-cash/max(total,1))*100:.0f}%")

        if pf.holdings:
            rows_pf = []
            for h in pf.holdings:
                cp = _price(h.code, h.avg_cost)
                pnl = (cp - h.avg_cost) / h.avg_cost * 100
                rows_pf.append({
                    "代码": h.code, "名称": h.name, "份额": h.shares,
                    "成本价": f"{h.avg_cost:.4f}", "现价": f"{cp:.4f}",
                    "盈亏": f"{pnl:+.1f}%",
                })
            df_pf = pd.DataFrame(rows_pf)
            st.dataframe(df_pf.style.map(
                lambda v: f"color: {'red' if v.startswith('-') else 'green'}; font-weight: bold;"
                if isinstance(v, str) and v.endswith('%') else "",
                subset=["盈亏"]
            ), use_container_width=True, hide_index=True)
        else:
            st.info("暂无持仓。运行 `python etf-agent.py` 后自动交易。")
    except Exception as e:
        st.info(f"持仓数据暂不可用: {e}")

    # ── 模拟交易策略表现 ──
    try:
        from autotrade import PERF_LOG
        import json
        if PERF_LOG.exists():
            with open(PERF_LOG, "r", encoding="utf-8") as f:
                perf = json.load(f)
            st.subheader("模拟交易策略表现")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总收益率", f"{perf.get('total_return_pct', 0):+.2f}%")
            c2.metric("最大回撤", f"{perf.get('max_drawdown_pct', 0):.2f}%")
            c3.metric("夏普比率", perf.get("sharpe_ratio", "N/A"))
            c4.metric("当前总值", f"{perf.get('current_value', 0):.0f}")

            from autotrade import TRADE_LOG
            if TRADE_LOG.exists():
                with open(TRADE_LOG, "r", encoding="utf-8") as f:
                    snaps = json.load(f)
                if len(snaps) > 1:
                    df_perf = pd.DataFrame([
                        {"日期": s["date"], "总资产": s["total"]}
                        for s in snaps
                    ])
                    st.line_chart(df_perf.set_index("日期"))
    except Exception:
        pass

    # ── 运行提示 ──
    st.sidebar.divider()
    st.sidebar.info(
        "**启动命令**\n\n"
        "```\nstreamlit run dashboard.py\n```\n\n"
        "数据来源: `data/snapshots/*.json`"
    )


if __name__ == "__main__":
    main()
