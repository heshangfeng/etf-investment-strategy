"""早盘全量分析推送脚本。
8:00 运行全量分析 -> 推送过滤后的操盘建议到手机。
SignalCaliberFilter 已在 etf-agent.py 中提前执行。
"""
import os, sys, json, urllib.request, subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def push(text: str):
    env_file = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_file):
        return
    key = ""
    for line in open(env_file, "r", encoding="utf-8"):
        if line.startswith("PUSHDEER_KEY="):
            key = line.split("=", 1)[1].strip()
            break
    if not key:
        return
    body = json.dumps({"pushkey": key, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        "https://api2.pushdeer.com/message/push", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8")).get("code") == 0


def main():
    log_file = os.path.join(BASE_DIR, "logs", f"{datetime.now().strftime('%Y%m%d')}_morning.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} - {msg}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {msg}\n")

    log("=== 早盘全量分析 Start ===")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "etf-agent.py"], cwd=BASE_DIR,
        capture_output=True, text=True, encoding="utf-8", timeout=600, env=env,
    )
    if r.returncode != 0:
        log(f"失败 (code {r.returncode})")
        push(f"ETF分析失败\n{r.stderr[:500]}")
        return
    log("完成")

    out = r.stdout + r.stderr
    lines = []
    for line in out.split("\n"):
        if any(kw in line for kw in ["🚫", "✅ T", "建议持仓", "模拟调仓",
                                       "集中度预警", "🎯", "尾部风险"]):
            lines.append(line.strip())

    text = "\n".join(lines[:30]) if lines else out[:1500]
    push(f"🏆 ETF操盘建议 {datetime.now().strftime('%m/%d')}\n\n{text}")
    log("推送完成")
    log("=== 早盘全量分析 End ===")


if __name__ == "__main__":
    main()
