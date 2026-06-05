"""
ETF 智能投研看板 - Streamlit Dashboard

启动: streamlit run dashboard.py
"""
import json
import glob
import os


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from infra.logger import get_logger; logger = get_logger(__name__)

# ── 页面配置 ──
st.set_page_config(page_title="ETF 智能投研看板", layout="wide")

from core.config import OUTPUT_DIR, SNAPSHOT_DIR, REVIEW_DIR, FEE_RATE
from pathlib import Path
SNAPSHOT_DIR = Path(SNAPSHOT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
REVIEW_DIR = Path(REVIEW_DIR)
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


# ── 个性化建议（根据实际持仓调整） ──

def _personalize_operations(etfs):
    """
    根据实际持仓调整每只ETF的操作/仓位建议。
    规则:
      1. 系统建议减持,用户不持有 → 保持原样（通用建议）
      2. 系统建议减持,用户持有且亏损 → 用投研分析判断割肉还是观察:
         - 共识"存在分歧"/"严重分歧"且评分靠近中性区(≥45) → 观察（信号不统一，不急于割）
         - Agent数<6（LLM覆盖不足）且评分在35-50之间 → 观察（数据不足，信号弱）
         - 其他情况 → 保持减持（信号明确，割肉合理）
      3. 系统建议持有/长期持有,用户不持有 → 改为买入开仓
    """
    try:
        from trading.portfolio import load, _price
        pf = load()
        holdings = {h.code: h for h in pf.holdings}
    except Exception:
        holdings = {}

    # 计算组合总资产用于实际仓位比例
    try:
        pf_total = pf.cash + sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
    except Exception:
        pf_total = 0

    for e in etfs:
        code = e["code"]
        op = e.get("operation", "")
        e["_suggested_pos"] = e.get("position_pct", 0)  # 保留系统建议仓位

        if code in holdings:
            h = holdings[code]
            cp = _price(code, h.avg_cost)
            pnl_pct = (cp - h.avg_cost) / h.avg_cost * 100
            e["_held"] = True
            e["_pnl_pct"] = round(pnl_pct, 2)
            actual_pos = (h.shares * cp) / pf_total if pf_total > 0 else 0
            e["position_pct"] = round(actual_pos, 4)

            # 规则2: 系统说减持且亏损 → 投研分析决定是否观察
            if op == "减持" and pnl_pct < 0:
                score = e.get("final_score", 50)
                consensus = e.get("consensus", "")
                n_agents = len(e.get("agents", []))
                rating = e.get("final_rating", "")
                weak_signal = (
                    (consensus in ("存在分歧", "严重分歧") and score >= 45) or
                    (n_agents < 6 and 35 <= score < 50)
                )
                if rating == "强烈看空":
                    pass
                elif weak_signal:
                    e["operation"] = "观察"
                    e["holding_period"] = "短期观察"
                    e["_personalized"] = True
        else:
            e["_held"] = False
            e["position_pct"] = 0.0
            if op in ("减持", "卖出", "强烈卖出"):
                e["operation"] = "不操作"
                e["_personalized"] = True
            elif op in ("持有", "长期持有"):
                # 系统评级看空/强烈看空时，即使操作是持有也不建议买入
                rating = e.get("final_rating", "")
                if rating in ("看空", "强烈看空"):
                    e["operation"] = "不操作"
                else:
                    e["operation"] = "买入"
                    e["holding_period"] = "中期(1-3月)"
                e["_personalized"] = True

    # 最终清理：统一 _suggested_pos
    for e in etfs:
        op = e.get("operation", "")
        # 未持仓且非买入操作 → 不操作
        if not e.get("_held") and op not in ("不操作", "买入"):
            e["operation"] = "不操作"
            e["_personalized"] = True
            op = "不操作"

        # 根据操作更新建议仓位
        if op == "不操作":
            e["_suggested_pos"] = 0.0
        elif op == "观察":
            e["_suggested_pos"] = e.get("position_pct", 0)  # 保持现有仓位
        elif op == "买入":
            rating = e.get("final_rating", "")
            if rating in ("看空", "强烈看空"):
                e["_suggested_pos"] = 0.0
            else:
                # 以评分为主，评级为辅
                # 评分≥65 或 评级看多/强烈看多 → 7.5%
                # 评分50-64 → 5%（即使评级看多也只给5%，如电力ETF）
                score = e.get("final_score", 50)
                if score >= 80:
                    e["_suggested_pos"] = 0.10
                elif score >= 65 or (rating in ("看多", "强烈看多") and score >= 50):
                    e["_suggested_pos"] = 0.075
                elif score >= 50:
                    e["_suggested_pos"] = 0.05
                elif score >= 35:
                    e["_suggested_pos"] = 0.03
                else:
                    e["_suggested_pos"] = 0.02
        elif op in ("持有", "长期持有"):
            e["_suggested_pos"] = e.get("position_pct", 0)  # 仓位不动
        elif op in ("减持",):
            # 减持 = 减到当前仓位的一半（非清仓）
            pos = e.get("position_pct", 0)
            e["_suggested_pos"] = round(pos * 0.5, 4)
        elif op in ("卖出", "强烈卖出"):
            e["_suggested_pos"] = 0.0  # 强烈看空 → 清仓


# ── 主界面 ──

def main():
    import json
    data = load_data()
    if data is None:
        st.warning("暂无数据。请先运行 etf-agent.py 生成报告。")
        st.info(f"期望快照目录: {SNAPSHOT_DIR}")
        return

    etfs = data["etfs"]
    _personalize_operations(etfs)
    date_str = data["date"]
    st.title(f"📊 ETF 智能投研看板 · {date_str}")

    # ── 顶部指标行 ──
    avg_score = sum(e["final_score"] for e in etfs) / len(etfs) if etfs else 0
    actual_total = sum(e.get("position_pct", 0) for e in etfs)
    sugg_total = sum(e.get("_suggested_pos", 0) for e in etfs)
    # 个性化后的操作计数
    buy_ops = sum(1 for e in etfs if e.get("operation") in ("买入", "强烈买入", "增持"))
    hold_ops = sum(1 for e in etfs if e.get("operation") in ("持有", "长期持有", "观察"))
    reduce_ops = sum(1 for e in etfs if e.get("operation") in ("减持", "卖出", "强烈卖出"))
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("ETF 总数", len(etfs))
    col2.metric("平均得分", f"{avg_score:.1f}")
    col3.metric("实际总仓位", f"{actual_total*100:.1f}%")
    col4.metric("建议总仓位", f"{sugg_total*100:.1f}%")
    col5.metric("建议买入", buy_ops)
    col6.metric("建议持有/观察", hold_ops)
    col7.metric("建议减持/卖出", reduce_ops)

    # ── 筛选 ──
    types = list(set(e.get("type", "未知") for e in etfs))
    selected_type = st.sidebar.selectbox("板块类型", ["全部"] + sorted(types))
    filtered = [e for e in etfs if selected_type == "全部" or e.get("type") == selected_type]

    # ── ETF 表格 ──
    st.subheader("ETF 持仓表")
    rows = []
    for e in filtered:
        op = e.get("operation", "")
        personalized = e.get("_personalized", False)
        if personalized:
            op_display = op
        else:
            op_display = op
        rows.append({
            "代码": e["code"],
            "名称": e["name"],
            "类型": e.get("type", ""),
            "评级": e.get("final_rating", ""),
            "得分": e["final_score"],
            "实际仓位": f"{e.get('position_pct', 0)*100:.1f}%",
            "建议仓位": f"{e.get('_suggested_pos', 0)*100:.1f}%",
            "操作": op_display,
            "持有周期": e.get("holding_period", ""),
            "共识度": e.get("consensus", ""),
            "现价": e.get("close_price", ""),
        })
    df = pd.DataFrame(rows)
    df.index = df["代码"]  # 用代码做索引方便选中

    # 操作列颜色映射
    op_colors = {
        "强烈买入": "color: #e74c3c; font-weight: bold;",
        "买入": "color: #e67e22; font-weight: bold;",
        "增持": "color: #e67e22; font-weight: bold;",
        "持有": "color: #3498db;",
        "长期持有": "color: #2980b9;",
        "观察": "color: #f39c12; font-weight: bold;",
        "减持": "color: #27ae60;",
        "卖出": "color: #2ecc71; font-weight: bold;",
        "强烈卖出": "color: #1abc9c; font-weight: bold;",
    }
    cell_hover = {"selector": "td:hover", "props": "background-color: #ffffcc;"}
    styled = df.style.map(
        lambda v: f"color: {rating_color(v)}; font-weight: bold;" if v in ("强烈看多", "看多", "中性", "看空", "强烈看空") else "",
        subset=["评级"]
    ).map(
        lambda v: f"color: {score_color(float(v.rstrip('%')))};" if isinstance(v, str) and v.endswith('%') else "",
        subset=["实际仓位"]
    ).map(
        lambda v: f"color: #95a5a6;" if isinstance(v, str) and v.endswith('%') else "",
        subset=["建议仓位"]
    ).map(
        lambda v: op_colors.get(v, ""),
        subset=["操作"]
    )
    st.dataframe(styled, width="stretch", height=min(60 + len(df) * 35, 600))

    # ── 评级分布图 ──
    rating_counts = df["评级"].value_counts()
    fig_ratings = px.bar(
        x=rating_counts.index, y=rating_counts.values,
        title="评级分布",
        labels={"x": "评级", "y": "只数"},
        color=rating_counts.index,
        color_discrete_map={
            "强烈看多": "#e74c3c", "看多": "#e67e22", "中性": "#f1c40f",
            "看空": "#27ae60", "强烈看空": "#2ecc71",
        },
        text=rating_counts.values,
    )
    fig_ratings.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig_ratings, width='stretch')

    # ── 板块分布饼图 ──
    type_counts = df["类型"].value_counts()
    fig_types = px.pie(
        values=type_counts.values, names=type_counts.index,
        title="板块分布",
        hole=0.4,
    )
    fig_types.update_layout(height=280)
    st.plotly_chart(fig_types, width='stretch')

    # ── 评分分布直方图 ──
    fig_scores = px.histogram(
        df, x="得分", nbins=20,
        title="评分分布",
        labels={"得分": "评分区间", "count": "ETF数量"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_scores.update_layout(height=250)
    st.plotly_chart(fig_scores, width='stretch')

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

                # Agent 评分雷达图
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=df_a["得分"].values,
                    theta=df_a["智能体"].values,
                    fill="toself",
                    line_color="#1f77b4",
                    marker_color="#1f77b4",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    height=350,
                    title="Agent 评分雷达",
                    margin=dict(l=80, r=80, t=40, b=40),
                )
                st.plotly_chart(fig_radar, width='stretch')

    # ── 底部统计 ──
    st.divider()
    col_x, col_y = st.columns(2)
    with col_x:
        st.caption(f"历史快照: {data['snap_count']} 天")
    with col_y:
        cum = data.get("cum_stats", {})
        if cum:
            st.caption(f"累计复盘: {cum.get('total_runs', 0)} 次 | "
                       f"准确率: {cum.get('overall_accuracy_pct', 0):.1f}%")

    # ── 投资组合面板（实际持仓） ──
    st.divider()
    st.subheader("我的投资组合（实际持仓）")
    st.caption("顶部[建议总仓位]=系统推荐 ｜ 此处[实际仓位]=你的真实持仓比例, 自动交易后两者应基本一致")
    try:
        import json
        from trading.portfolio import load, _price, _price_detail
        from trading.autotrade import PERF_LOG
        pf = load()
        cash = pf.cash
        holdings_cost = sum(h.shares * h.avg_cost for h in pf.holdings) if pf.holdings else 0
        market_value = sum(h.shares * _price(h.code, h.avg_cost) for h in pf.holdings)
        total = cash + market_value
        pnl_amount = market_value - holdings_cost

        # 从performance.json读取累计/日收益
        _perf_data = {}
        if PERF_LOG.exists():
            with open(PERF_LOG, "r", encoding="utf-8") as _pf:
                _perf_data = json.load(_pf)
        daily_pnl = _perf_data.get("daily_pnl", 0)
        cum_pnl = _perf_data.get("cumulative_pnl", total - 670000)

        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
        col_a.metric("总资产", f"{total:.0f}")
        col_b.metric("持仓市值", f"{market_value:.0f}")
        col_c.metric("实际仓位", f"{(1-cash/max(total,1))*100:.0f}%")
        col_d.metric("今日收益", f"{daily_pnl:+.0f}", delta_color="off" if daily_pnl >= 0 else "inverse")
        col_e.metric("持仓收益", f"{pnl_amount:+.0f}", delta_color="off" if pnl_amount >= 0 else "inverse")
        col_f.metric("累计收益", f"{cum_pnl:+.0f}", delta_color="off" if cum_pnl >= 0 else "inverse")

        if pf.holdings:
            rows_pf = []
            for h in pf.holdings:
                cp, prev_close = _price_detail(h.code, h.avg_cost)
                pnl_pct = (cp - h.avg_cost) / h.avg_cost * 100
                pnl_amt = (cp - h.avg_cost) * h.shares
                today_chg = (cp - prev_close) * h.shares
                mkt_val = h.shares * cp
                rows_pf.append({
                    "代码": h.code, "名称": h.name,
                    "成本价": f"{h.avg_cost:.4f}", "现价": f"{cp:.4f}",
                    "持仓市值": f"{mkt_val:.0f}",
                    "今日收益": f"{today_chg:+.0f}",
                    "持仓收益率": f"{pnl_pct:+.2f}%",
                    "持仓收益": f"{pnl_amt:+.0f}",
                })
            df_pf = pd.DataFrame(rows_pf)
            st.dataframe(df_pf.style.map(
                lambda v: f"color: {'red' if v.startswith('-') else 'green'}; font-weight: bold;"
                if isinstance(v, str) and (v.endswith('%') or v.startswith('+') or v.startswith('-')) and v[0] in '+-' else "",
                subset=["今日收益", "持仓收益率", "持仓收益"]
            ), width='stretch', hide_index=True)
        else:
            st.info("暂无持仓。运行 `python etf-agent.py` 后自动交易。")

        # ── 月度汇总 ──
        try:
            from trading.autotrade import TRADE_LOG, PERF_LOG
            if TRADE_LOG.exists():
                with open(TRADE_LOG, "r", encoding="utf-8") as _f:
                    _snaps = json.load(_f)
                with open(PERF_LOG, "r", encoding="utf-8") as _f:
                    _perf = json.load(_f)
                if _snaps:
                    _monthly = {}
                    for _s in _snaps:
                        _m = _s["date"][:7]
                        if _m not in _monthly:
                            _monthly[_m] = {"first": _s, "last": _s}
                        else:
                            if _s["date"] < _monthly[_m]["first"]["date"]:
                                _monthly[_m]["first"] = _s
                            if _s["date"] > _monthly[_m]["last"]["date"]:
                                _monthly[_m]["last"] = _s
                    # 加上当月最新值（即使今日无交易）
                    _today_m = __import__("datetime").datetime.now().strftime("%Y-%m")
                    if _today_m in _monthly:
                        _monthly[_today_m]["last"] = {
                            "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
                            "total": _perf.get("current_value", _monthly[_today_m]["last"]["total"]),
                        }
                    _month_rows = []
                    for _m in sorted(_monthly.keys(), reverse=True):
                        _first = _monthly[_m]["first"]
                        _last = _monthly[_m]["last"]
                        _start = _first["total"]
                        _end = _last["total"]
                        _pnl = _end - _start
                        _ret = (_end / _start - 1) * 100
                        _month_rows.append({
                            "月份": _m, "月初总值": f"{_start:.0f}",
                            "月末总值": f"{_end:.0f}",
                            "月度盈亏": f"{_pnl:+.0f}",
                            "月度收益率": f"{_ret:+.2f}%",
                        })
                    if _month_rows:
                        st.subheader("月度收益汇总")
                        _df_m = pd.DataFrame(_month_rows)
                        st.dataframe(_df_m.style.map(
                            lambda v: f"color: {'red' if v.startswith('-') else 'green'}; font-weight: bold;"
                            if isinstance(v, str) and (v.endswith('%') or (v.startswith('+') or v.startswith('-'))) and not v.startswith('2') else "",
                            subset=["月度盈亏", "月度收益率"]
                        ), width="stretch", hide_index=True)
        except Exception:
            pass
    except Exception as e:
        st.info(f"持仓数据暂不可用: {e}")

    # ── 模拟交易策略表现 ──
    try:
        from trading.autotrade import PERF_LOG
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

            from trading.autotrade import TRADE_LOG
            if TRADE_LOG.exists():
                with open(TRADE_LOG, "r", encoding="utf-8") as f:
                    snaps = json.load(f)
                if len(snaps) > 1:
                    df_perf = pd.DataFrame([
                        {"date": s["date"], "total": s["total"]}
                        for s in snaps
                    ])
                    fig_equity = go.Figure(data=go.Scatter(
                        x=df_perf["date"], y=df_perf["total"],
                        mode="lines+markers",
                        line=dict(color="#1f77b4", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(31, 119, 180, 0.1)",
                    ))
                    fig_equity.update_layout(
                        title="资产曲线",
                        xaxis_title="日期",
                        yaxis_title="总资产",
                        height=350,
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                    st.plotly_chart(fig_equity, width='stretch')
    except Exception:
        logger.warning("加载模拟交易策略表现失败", exc_info=True)
        pass

    # ── 历史交易记录 ──
    st.divider()
    st.subheader("历史交易记录")
    try:
        from trading.autotrade import TRADE_LOG
        if TRADE_LOG.exists():
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                tl_snaps = json.load(f)
            for snap in reversed(tl_snaps):
                trades = snap.get("trades", {})
                date = snap.get("date", "?")
                buys = trades.get("buys", [])
                sells = trades.get("sells", [])
                if not buys and not sells:
                    continue

                with st.expander(f"📅 {date}  |  总资产 {snap.get('total', 0):.0f}", expanded=False):
                    if sells:
                        sell_rows = []
                        for t in sells:
                            fee = t.get("proceeds", 0) * FEE_RATE
                            net = t["proceeds"] - fee
                            sell_rows.append({
                                "方向": "🔴 卖出",
                                "代码": t["code"],
                                "名称": t["name"],
                                "份额": t["shares"],
                                "价格": f"{t['price']:.4f}",
                                "金额": f"{t['proceeds']:.0f}",
                                "佣金": f"{fee:.1f}",
                                "净到账": f"{net:.0f}",
                            })
                        df_sell = pd.DataFrame(sell_rows)
                        st.dataframe(df_sell, width="stretch", hide_index=True)

                    if buys:
                        buy_rows = []
                        for t in buys:
                            fee = t.get("cost", 0) * FEE_RATE
                            net = t["cost"] + fee
                            buy_rows.append({
                                "方向": "🟢 买入",
                                "代码": t["code"],
                                "名称": t["name"],
                                "份额": t["shares"],
                                "价格": f"{t['price']:.4f}",
                                "金额": f"{t['cost']:.0f}",
                                "佣金": f"{fee:.1f}",
                                "实付": f"{net:.0f}",
                            })
                        df_buy = pd.DataFrame(buy_rows)
                        st.dataframe(df_buy, width="stretch", hide_index=True)

                    # Show balance change for this day
                    cash_change = snap.get("cash", 0)
                    st.caption(f"当日现金: {cash_change:.0f}  |  持仓市值: {snap.get('market_value', 0):.0f}")
    except Exception as e:
        st.caption(f"交易记录暂不可用: {e}")

    # ── 运行提示 ──
    st.sidebar.divider()
    st.sidebar.info(
        "**运行命令**\n\n"
        "前台（终端关即停）：\n"
        "```powershell\npowershell -File start_dashboard.ps1\n```\n"
        "后台（关终端不影响）：\n"
        "```powershell\nStart-Process -WindowStyle Hidden -FilePath \"powershell\" -ArgumentList \"-File start_dashboard.ps1\"\n```\n\n"
        "数据来源: `data/snapshots/*.json`"
    )


if __name__ == "__main__":
    main()

