"""早盘全量分析推送脚本。
8:00 运行：全量分析 → 推送操盘建议到手机（不执行交易）。
开盘后由 execute_open.py 用开盘价执行模拟交易。
"""
import os
import subprocess
import sys
import json
import urllib.request
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
PUSH_URL = "https://api2.pushdeer.com/message/push"


def get_pushdeer_key() -> str:
    if not os.path.isfile(ENV_FILE):
        return ""
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("PUSHDEER_KEY="):
                return line.split("=", 1)[1]
    return ""


def push(text: str):
    key = get_pushdeer_key()
    if not key:
        print("无 PUSHDEER_KEY，跳过推送")
        print(text)
        return
    body = json.dumps({"pushkey": key, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        PUSH_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))
        if resp_data.get("code") == 0:
            print("推送成功")
        else:
            print(f"推送返回异常: {resp_data}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}_morning.log")

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - {msg}\n"
        print(line, end="")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

    log("=== 早盘全量分析 Start ===")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "etf-agent.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=600,
    )

    if result.returncode != 0:
        log(f"etf-agent.py failed (code {result.returncode})")
        push(f"ETF早盘分析失败\n{result.stderr[:500]}")
        return

    log("etf-agent.py completed successfully")

    # 提取操盘建议部分（从输出中截取关键内容）
    output = result.stdout + result.stderr
    # 只推送建议部分（去掉进度条等噪音）
    lines = output.split("\n")
    relevant = []
    for line in lines:
        if any(kw in line for kw in ["✅ T", "建议持仓", "操作建议", "建议买入",
                                       "建议卖出", "继续持有", "集中度预警",
                                       "止损止盈", "准确率", "🎯", "⚠️"]):
            relevant.append(line.strip())

    text = "\n".join(relevant[:30]) if relevant else output[:1500]
    push(f"🏆 ETF操盘建议 {datetime.now().strftime('%m/%d')}\n\n{text}")
    log(f"推送完成 ({len(text)} chars)")
    log("=== 早盘全量分析 End ===")


if __name__ == "__main__":
    main()
