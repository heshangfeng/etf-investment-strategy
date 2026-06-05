"""
开盘模拟执行脚本。9:35 运行：
读取 _trade_decisions.json（已过滤），用开盘价执行交易，推送结果。
"""
import os, sys, json, urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DECISIONS_FILE = BASE_DIR / "data" / "_trade_decisions.json"
PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio.json"
PUSH_URL = "https://api2.pushdeer.com/message/push"
FEE_RATE = 0.0003


def log(msg: str):
    os.makedirs(BASE_DIR / "logs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} - {msg}")
    with open(BASE_DIR / "logs" / f"{datetime.now().strftime('%Y%m%d')}_open_exec.log", "a", encoding="utf-8") as f:
        f.write(f"{ts} - {msg}\n")


def push(text: str):
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    key = ""
    for line in open(env_file, "r", encoding="utf-8"):
        if line.startswith("PUSHDEER_KEY="):
            key = line.split("=", 1)[1].strip()
            break
    if not key:
        return
    data = json.dumps({"pushkey": key, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        PUSH_URL, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        r = json.loads(resp.read().decode("utf-8"))
        return r.get("code") == 0


def get_open_price(code: str) -> float:
    try:
        sys.path.insert(0, str(BASE_DIR))
        from core.data import DataCollectAgent
        df = DataCollectAgent.get_etf_price(code)
        if df is not None and len(df) > 0:
            return float(df["open"].iloc[-1])
    except Exception:
        pass
    return 0.0


def main():
    log("=== 开盘模拟执行 Start ===")
    if not DECISIONS_FILE.exists():
        log(f"决策文件不存在，请先运行 etf-agent.py")
        return
    decisions = json.load(open(DECISIONS_FILE, "r", encoding="utf-8"))
    log(f"加载 {len(decisions)} 条决策")

    pf = {"cash": 670000.0, "holdings": [], "transactions": []}
    if PORTFOLIO_FILE.exists():
        pf = json.load(open(PORTFOLIO_FILE, "r", encoding="utf-8"))

    lines = [f"🏆 开盘执行 {datetime.now().strftime('%m/%d %H:%M')}\n"]

    for d in decisions:
        code = d["code"]
        name = d.get("name", code)
        action = d["action"]

        if action == "买入":
            price = get_open_price(code)
            if price <= 0:
                log(f"  ⚠️ {name} 无开盘价，跳过")
                continue
            alloc = pf["cash"] * d.get("position", 0.05)
            shares = int(alloc / price / 100) * 100
            if shares < 100 or shares * price > pf["cash"]:
                continue
            cost = shares * price
            fee = cost * FEE_RATE
            pf["cash"] -= (cost + fee)
            pf["holdings"].append({
                "code": code, "name": name, "shares": shares,
                "avg_cost": round(price, 4), "added": datetime.now().strftime("%Y-%m-%d"),
            })
            lines.append(f"📗 买入 {name} {shares}份 @ {price:.4f}")
            log(f"  买入 {name} {shares}份 @ {price:.4f}")

        elif action in ("卖出", "减持"):
            price = get_open_price(code)
            if price <= 0:
                continue
            for h in list(pf["holdings"]):
                if h["code"] == code:
                    sell = h["shares"] if action == "卖出" else max(int(h["shares"] * 0.5 / 100) * 100, 0)
                    if sell >= 100:
                        proceeds = sell * price
                        fee = proceeds * FEE_RATE
                        pf["cash"] += (proceeds - fee)
                        h["shares"] -= sell
                        lines.append(f"📕 {action} {name} {sell}份 @ {price:.4f}")
                        log(f"  {action} {name} {sell}份 @ {price:.4f}")
                    if h["shares"] <= 0:
                        pf["holdings"].remove(h)
                    break

        elif action in ("跳过卖出", "跳过买入"):
            conv = d.get("conviction", 0)
            lines.append(f"⏭️ 跳过 {name} (可信度{conv:.0f})")
            log(f"  跳过 {name} (可信度{conv:.0f})")

        elif action == "持有":
            log(f"  持有 {name}")

    json.dump(pf, open(PORTFOLIO_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    push("\n".join(lines))
    log("推送成功")
    log("=== 开盘模拟执行 End ===")


if __name__ == "__main__":
    main()
