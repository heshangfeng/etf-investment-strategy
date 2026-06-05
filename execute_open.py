"""
开盘模拟执行脚本。9:35 运行：
1. 加载早上 etf-agent.py 保存的分析报告
2. 用开盘价执行 auto_trade（模拟你开盘后的操作）
3. 推送执行结果到手机
"""
import os
import sys
import pickle
import json
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
ENV_FILE = BASE_DIR / ".env"
REPORT_PKL = BASE_DIR / "data" / "_last_reports.pkl"
PUSH_URL = "https://api2.pushdeer.com/message/push"


def log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - {msg}\n"
    print(line, end="")
    with open(LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}_open_exec.log", "a", encoding="utf-8") as f:
        f.write(line)


def get_pushdeer_key() -> str:
    if not ENV_FILE.exists():
        return ""
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("PUSHDEER_KEY="):
                return line.split("=", 1)[1]
    return ""


def push(title: str, body: str):
    key = get_pushdeer_key()
    if not key:
        log("无 PUSHDEER_KEY，跳过推送")
        return
    text = f"{title}\n\n{body}"
    try:
        data = json.dumps({"pushkey": key, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            PUSH_URL, data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            r = json.loads(resp.read().decode("utf-8"))
            if r.get("code") == 0:
                log("推送成功")
            else:
                log(f"推送返回: {r}")
    except Exception as e:
        log(f"推送失败: {e}")


def main():
    log("=== 开盘模拟执行 Start ===")

    # 加载分析报告
    if not REPORT_PKL.exists():
        log("未找到分析报告，请先运行 etf-agent.py")
        return
    with open(REPORT_PKL, "rb") as f:
        final_reports = pickle.load(f)
    log(f"加载报告: {len(final_reports)} 只 ETF")

    # 执行模拟交易（auto_trade 会调用 _price 获取实时价，此时有开盘价）
    try:
        from trading.autotrade import auto_trade
        trades = auto_trade(final_reports)
        b, s = len(trades["buys"]), len(trades["sells"])
        log(f"模拟交易完成: 买入{b}只, 卖出{s}只")
    except Exception as e:
        log(f"模拟交易失败: {e}")
        push("ETF开盘执行失败", f"自动交易异常: {e}")
        return

    # 推送执行结果
    lines = [f"🏆 开盘模拟执行 {datetime.now().strftime('%m/%d %H:%M')}\n"]
    for t in trades.get("buys", []):
        lines.append(f"📗 买入 {t['name']}: {t['shares']}份 @ {t['price']:.4f}")
    for t in trades.get("sells", []):
        lines.append(f"📕 卖出 {t['name']}: {t['shares']}份 @ {t['price']:.4f}")
    for t in trades.get("decisions", []):
        if t["action"] in ("跳过卖出", "跳过买入"):
            conv = t.get("conviction", 0)
            lines.append(f"⏭️ {t['action']} {t['name']} (可信度{conv:.0f})")
    if not any(trades.values()):
        lines.append("无交易操作")
    push("开盘执行结果", "\n".join(lines))

    log("=== 开盘模拟执行 End ===")


if __name__ == "__main__":
    main()
