"""
ETF 智能投资分析系统 - 复盘引擎
"""
import json
import os
import glob
import numpy as np
from datetime import datetime
import pandas as pd
from math import comb, log, exp

from config import ETF_POOL, RATING_ORDER, SNAPSHOT_DIR as CFG_SNAPSHOT, REVIEW_DIR as CFG_REVIEW
from data import DataCollectAgent
from models import FinalResearchReport, AgentReport


# ====================== 【复盘引擎】 ======================
class ReviewManager:
    """复盘引擎：保存每日快照、T+1验证操作建议、跟踪累计准确率"""

    SNAPSHOT_DIR = CFG_SNAPSHOT
    REVIEW_DIR_PATH = CFG_REVIEW
    REVIEW_FILE = f"{CFG_REVIEW}/cumulative_stats.json"

    @classmethod
    def process(cls, all_reports: list[FinalResearchReport]):
        """执行完整复盘流程：保存今日快照 → T+1验证 → 更新累计统计 → 打印"""
        os.makedirs(f"{cls.SNAPSHOT_DIR}", exist_ok=True)
        os.makedirs(cls.REVIEW_DIR_PATH, exist_ok=True)

        # 1. 保存今日快照
        cls._save_snapshot(all_reports)

        # 2. T+1验证昨日建议
        verifications = cls._run_t1_verification()
        if not verifications:
            return  # 首次运行，无历史数据

        # 3. 更新累计统计
        cls._update_cumulative_stats(verifications)

        # 4. 打印复盘摘要
        cls._print_review_summary()

        # 4b. 校准曲线
        cls._print_calibration_curve()

        # 5. 统计显著性检验
        cls._compute_significance()

        # 6. 数据新鲜度校验
        cls._check_data_freshness()

        # 7. 更新Agent权重文件
        cls._update_agent_weights()

    @classmethod
    def _save_snapshot(cls, all_reports: list[FinalResearchReport]):
        """保存今日操作建议快照"""
        today = datetime.now().strftime("%Y%m%d")
        etfs = []
        for fr in all_reports:
            agents_data = [{"n": r.agent_name, "rt": r.rating, "sc": r.score, "src": r.source[:4]}
                          for r in fr.agent_reports]
            # 从缓存获取收盘价
            try:
                df = DataCollectAgent.get_etf_price(fr.etf_info['code'])
                close_px = float(df["close"].iloc[-1])
            except Exception:
                close_px = 0.0

            etfs.append({
                "code": fr.etf_info['code'], "name": fr.etf_info['name'], "type": fr.etf_info['type'],
                "operation": fr.operation, "holding_period": fr.holding_period,
                "final_score": fr.final_score, "final_rating": fr.final_rating,
                "position_pct": fr.suggested_position_pct, "consensus": fr.consensus_level,
                "close_price": close_px,
                "agents": agents_data
            })

        snapshot = {
            "date": today,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_volume_bn": DataCollectAgent.get_market_total_volume(),
            "etfs": etfs
        }
        path = f"{cls.SNAPSHOT_DIR}/{today}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    @classmethod
    def _get_prev_snapshot(cls) -> tuple[dict, str] | None:
        """获取最近一次历史快照"""
        files = sorted(glob.glob(f"{cls.SNAPSHOT_DIR}/*.json"))
        if len(files) < 2:
            return None  # 只有今日文件或没有
        prev_path = files[-2]  # 上一次运行的快照
        prev_date = os.path.basename(prev_path).replace(".json", "")
        with open(prev_path, "r", encoding="utf-8") as f:
            return json.load(f), prev_date

    @classmethod
    def _run_t1_verification(cls) -> list[dict]:
        """T+1验证：对比昨日操作建议与今日实际涨跌"""
        result = cls._get_prev_snapshot()
        if result is None:
            return []
        prev_data, prev_date = result

        verifications = []
        for etf in prev_data["etfs"]:
            code = etf["code"]
            prev_price = etf["close_price"]
            if prev_price == 0:
                continue

            # 获取今日实际价格（从缓存）
            try:
                df = DataCollectAgent.get_etf_price(code)
                curr_price = float(df["close"].iloc[-1])
            except Exception:
                continue

            ret_pct = (curr_price - prev_price) / prev_price * 100
            is_correct = cls._check_direction(etf["operation"], ret_pct)

            verifications.append({
                "date": prev_date, "code": code, "name": etf["name"],
                "operation": etf["operation"], "return_pct": round(ret_pct, 2),
                "prev_price": prev_price, "curr_price": curr_price,
                "correct": is_correct
            })

        return verifications

    @staticmethod
    def _check_direction(operation: str, ret_pct: float) -> str:
        """判断操作建议方向是否正确"""
        buy_ops = {"强烈买入", "买入"}
        sell_ops = {"卖出", "强烈卖出"}
        hold_ops = {"长期持有", "持有"}

        if operation in buy_ops:
            if ret_pct > 1.0: return "正确"
            if ret_pct < -1.0: return "错误"
            return "持平"
        elif operation in sell_ops:
            if ret_pct < -1.0: return "正确"
            if ret_pct > 1.0: return "错误"
            return "持平"
        elif operation == "减持":
            if ret_pct < -1.0: return "正确"
            if ret_pct > 1.0: return "错误"
            return "持平"
        else:  # 持有/长期持有
            if -1.0 <= ret_pct <= 1.0: return "正确"
            if ret_pct > 1.0: return "错误"
            if ret_pct < -1.0: return "错误"
            return "正确"

    @classmethod
    def _update_cumulative_stats(cls, verifications: list[dict]):
        """更新累计复盘统计"""
        stats = {"last_date": "", "total_runs": 0, "total_verifications": 0,
                 "overall_accuracy_pct": 0.0, "by_operation": {}, "by_agent": {},
                 "by_date": [], "pending": []}

        # 加载已有统计
        if os.path.exists(cls.REVIEW_FILE):
            with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)

        today = datetime.now().strftime("%Y%m%d")

        # 统计本次验证结果
        correct = sum(1 for v in verifications if v["correct"] == "正确")
        wrong = sum(1 for v in verifications if v["correct"] == "错误")

        # 更新按操作类型
        for v in verifications:
            op = v["operation"]
            if op not in stats["by_operation"]:
                stats["by_operation"][op] = {"total": 0, "correct": 0, "wrong": 0}
            stats["by_operation"][op]["total"] += 1
            if v["correct"] == "正确":
                stats["by_operation"][op]["correct"] += 1
            elif v["correct"] == "错误":
                stats["by_operation"][op]["wrong"] += 1

        # 更新日期记录
        stats["by_date"].append({
            "d": verifications[0]["date"] if verifications else today,
            "verified": len(verifications),
            "correct": correct,
            "wrong": wrong,
            "pct": round(correct / (correct + wrong) * 100, 1) if (correct + wrong) > 0 else 0
        })

        # 更新Agent维度准确率（从最近快照读取agent评分方向 vs 实际涨跌）
        cls._update_agent_stats(stats)

        # 重新计算总体
        total_correct = sum(d["correct"] for d in stats["by_date"])
        total_wrong = sum(d["wrong"] for d in stats["by_date"])
        total_v = total_correct + total_wrong
        stats["overall_accuracy_pct"] = round(total_correct / total_v * 100, 1) if total_v > 0 else 0
        stats["total_verifications"] = total_v
        stats["total_runs"] = len(stats["by_date"])
        stats["last_date"] = today

        with open(cls.REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    @classmethod
    def _update_agent_stats(cls, stats: dict):
        """更新各智能体的方向准确率"""
        result = cls._get_prev_snapshot()
        if result is None:
            return
        prev_data, prev_date = result

        today_prices = {}  # 缓存今日价格
        for etf in prev_data["etfs"]:
            code = etf["code"]
            if code not in today_prices:
                try:
                    df = DataCollectAgent.get_etf_price(code)
                    today_prices[code] = float(df["close"].iloc[-1])
                except Exception:
                    continue

            prev_price = etf["close_price"]
            curr_price = today_prices[code]
            if prev_price == 0:
                continue
            actual_ret = (curr_price - prev_price) / prev_price * 100
            actual_dir = 1 if actual_ret > 0.3 else (-1 if actual_ret < -0.3 else 0)

            for a in etf["agents"]:
                aname = a["n"]
                if aname not in stats["by_agent"]:
                    stats["by_agent"][aname] = {"directional": 0, "hits": 0}

                rating_dir = cls._agent_direction(a["rt"])
                if rating_dir == 0 or actual_dir == 0:
                    continue  # 中性不参与方向统计

                stats["by_agent"][aname]["directional"] += 1
                if rating_dir == actual_dir:
                    stats["by_agent"][aname]["hits"] += 1

    @staticmethod
    def _agent_direction(rating: str) -> int:
        """将评级转为方向信号"""
        if rating in ("强烈看多", "看多"): return 1
        if rating in ("强烈看空", "看空"): return -1
        return 0

    @classmethod
    def _print_review_summary(cls):
        """打印复盘摘要到控制台"""
        if not os.path.exists(cls.REVIEW_FILE):
            return
        with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)

        print("\n" + "="*160)
        print("【📊 昨日复盘 | T+1验证】")
        print("="*160)

        if stats["by_date"]:
            last = stats["by_date"][-1]
            print(f"  验证日期: {last['d']} → {datetime.now().strftime('%Y%m%d')}")
            print(f"  验证标的: {last['verified']}只 | 正确: {last['correct']} | 错误: {last['wrong']} | 准确率: {last['pct']}%")

        print(f"\n  📈 累计复盘({stats['total_runs']}天/{stats['total_verifications']}次)")
        print(f"  整体准确率: {stats['overall_accuracy_pct']}%")

        # 按操作类型
        if stats["by_operation"]:
            print(f"\n  按操作类型:")
            for op, data in sorted(stats["by_operation"].items()):
                hit = data["correct"]
                tot = data["total"]
                p = round(hit / tot * 100, 1) if tot > 0 else 0
                bar = "█" * int(p / 10) + "░" * (10 - int(p / 10))
                print(f"    {op}: {bar} {p}% ({hit}/{tot})")

        # 按Agent（排序）
        if stats["by_agent"]:
            print(f"\n  智能体方向准确率排行:")
            sorted_agents = sorted(stats["by_agent"].items(), key=lambda x: x[1]["hits"]/max(x[1]["directional"],1), reverse=True)
            for rank, (name, data) in enumerate(sorted_agents, 1):
                d = data["directional"]
                h = data["hits"]
                p = round(h / d * 100, 1) if d > 0 else 0
                bar = "█" * int(p / 10) + "░" * (10 - int(p / 10))
                star = " ★" if rank == 1 else ""
                print(f"    {rank}. {name}: {bar} {p}% ({h}/{d}){star}")

        print()

    @classmethod
    def _print_calibration_curve(cls):
        """校准曲线：将评分分为十档，计算每档的实际平均收益"""
        files = sorted(glob.glob(f"{cls.SNAPSHOT_DIR}/*.json"))
        if len(files) < 2:
            return

        buckets = {i: {"returns": [], "count": 0} for i in range(0, 100, 10)}

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                snap = json.load(f)
            for etf in snap.get("etfs", []):
                score = etf.get("final_score", 50)
                px = etf.get("close_price", 0)
                if px == 0:
                    continue
                try:
                    df = DataCollectAgent.get_etf_price(etf["code"])
                except Exception:
                    continue
                sd = snap["date"]
                dl = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").tolist()
                if sd not in dl:
                    continue
                i = dl.index(sd)
                if i + 1 >= len(df):
                    continue
                ret = df.iloc[i + 1]["close"] / px - 1
                bucket = (int(score) // 10) * 10
                buckets[bucket]["returns"].append(ret)
                buckets[bucket]["count"] += 1

        total = sum(d["count"] for d in buckets.values())
        if total < 2:
            return

        all_rets = [r for d in buckets.values() for r in d["returns"]]
        overall = float(np.mean(all_rets)) * 100

        print("\n" + "="*80)
        print("【📊 校准曲线 | Calibration Curve】")
        print("="*80)
        print(f"  {'评分区间':<12} {'数量':<6} {'平均收益%':<10} {'校准偏差':<10}")
        print(f"  {'-'*38}")

        for bucket in range(0, 100, 10):
            d = buckets[bucket]
            if d["count"] == 0:
                continue
            avg = float(np.mean(d["returns"])) * 100
            bias = avg - overall
            print(f"  {bucket:3d}-{bucket+9:<6} {d['count']:<6} {avg:<+8.2f}%  {bias:<+8.2f}%")

        print(f"\n  总体均值: {overall:+.2f}%")
        print()

    @classmethod
    def _run_rolling_backtest(cls) -> list[dict]:
        """滚动回测：针对120+数据点的ETF，模拟20日滚动窗口方向准确率"""
        print("\n【滚动回测】检验趋势信号历史表现...")
        results = []
        pool = ETF_POOL
        for item in pool:
            code = item["code"]
            try:
                df = DataCollectAgent.get_etf_price(code)
                if df is None or len(df) < 120:
                    continue
                window_size = 20
                correct, total = 0, 0
                returns = []
                for i in range(len(df) - window_size - 1):
                    window = df.iloc[i:i + window_size]
                    curr_close = df.iloc[i + window_size]["close"]
                    next_close = df.iloc[i + window_size + 1]["close"]
                    ma5 = window["close"].tail(5).mean()
                    ma20 = window["close"].tail(20).mean()
                    signal = 1 if ma5 > ma20 else -1
                    actual_ret = (next_close - curr_close) / curr_close
                    actual_dir = 1 if actual_ret > 0 else -1
                    if signal == actual_dir:
                        correct += 1
                    total += 1
                    returns.append(actual_ret)
                if total == 0:
                    continue
                win_rate = correct / total * 100
                avg_ret = float(np.mean(returns)) * 100
                std_ret = float(np.std(returns)) * 100
                sharpe = (avg_ret / max(std_ret, 1e-6)) * (252 ** 0.5)
                results.append({
                    "code": code, "name": item["name"],
                    "windows": total, "win_rate": round(win_rate, 1),
                    "sharpe": round(sharpe, 2)
                })
            except Exception:
                continue
        if results:
            avg_win = float(np.mean([r["win_rate"] for r in results]))
            avg_sharpe = float(np.mean([r["sharpe"] for r in results]))
            print(f"  回测ETF: {len(results)}只 | 平均胜率: {avg_win:.1f}% | 平均夏普: {avg_sharpe:.2f}")
            for r in sorted(results, key=lambda x: x["win_rate"], reverse=True)[:5]:
                print(f"    {r['code']} {r['name']}: 胜率{r['win_rate']}% 夏普{r['sharpe']} (窗口{r['windows']})")
        else:
            print("  无满足条件的ETF（需120+数据点）")
        return results

    @classmethod
    def _compute_significance(cls):
        """二项检验：验证整体准确率是否显著高于随机(50%)"""
        if not os.path.exists(cls.REVIEW_FILE):
            return
        with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
        total = stats.get("total_verifications", 0)
        correct = sum(d["correct"] for d in stats.get("by_date", []))
        if total < 3:
            return
        p_null = 0.5
        if correct > total / 2:
            tail_sum = 0.0
            for k in range(correct, total + 1):
                log_p = log(comb(total, k)) + k * log(p_null) + (total - k) * log(1 - p_null)
                tail_sum += exp(log_p)
            p_value = min(tail_sum * 2, 1.0)
        else:
            tail_sum = 0.0
            for k in range(0, correct + 1):
                log_p = log(comb(total, k)) + k * log(p_null) + (total - k) * log(1 - p_null)
                tail_sum += exp(log_p)
            p_value = min(tail_sum * 2, 1.0)
        sig = "✅ 统计显著(p<0.05)" if p_value < 0.05 else "⚠️ 未达统计显著"
        print(f"\n【统计检验】二项检验: n={total} 正确={correct} p值={p_value:.4f} | {sig}")

    @classmethod
    def _check_data_freshness(cls):
        """校验今日ETF数据新鲜度"""
        print("\n【数据新鲜度校验】")
        pool = ETF_POOL[:5]
        today = datetime.now()
        stale, ok = 0, 0
        for item in pool:
            code = item["code"]
            try:
                df = DataCollectAgent.get_etf_price(code)
                if df is None or df.empty or "date" not in df.columns:
                    print(f"  ⚠️ {code} {item['name']}: 数据为空")
                    stale += 1
                    continue
                last_date = pd.to_datetime(df["date"].iloc[-1])
                days_diff = (today - last_date).days
                if days_diff > 5:
                    print(f"  ⚠️ {code} {item['name']}: 数据陈旧({days_diff}天前 {last_date.strftime('%Y-%m-%d')})")
                    stale += 1
                else:
                    print(f"  ✅ {code} {item['name']}: 最新{last_date.strftime('%Y-%m-%d')}({days_diff}天前)")
                    ok += 1
            except Exception as e:
                print(f"  ❌ {code} {item['name']}: 获取异常 → {str(e)[:60]}")
                stale += 1
        total_checked = ok + stale
        if total_checked > 0 and stale > total_checked / 2:
            print(f"  ⚠️ 警告: {stale}/{total_checked} 数据异常，结果可能不准确")
        else:
            print(f"  数据新鲜度: {ok}/{total_checked} 正常")

    @classmethod
    def _update_agent_weights(cls):
        """持久化Agent权重数据供ChiefDecisionAgent.load_agent_weights()读取"""
        stats = {"last_date": "", "total_runs": 0, "total_verifications": 0,
                 "overall_accuracy_pct": 0.0, "by_operation": {}, "by_agent": {},
                 "by_date": [], "pending": []}
        if os.path.exists(cls.REVIEW_FILE):
            with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        if not stats.get("by_agent"):
            cls._update_agent_stats(stats)
        with open(cls.REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
