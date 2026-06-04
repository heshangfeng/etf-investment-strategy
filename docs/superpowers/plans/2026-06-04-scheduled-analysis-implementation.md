# 定时分析推送系统 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增早盘8:15增量分析推送 + 增强午后15:30复盘推送内容

**Architecture:** 新建 `morning_update.py` 实现晨盘规则评分（复用`ChiefDecisionAgent`的数学逻辑，不调LLM），新建 `run_morning.ps1` 编排推送；修改 `autotrade.py` 增加模拟盘日盈亏字段，修改 `run_daily.ps1` 推送段追加当日盈亏。

**Tech Stack:** Python, PowerShell, PushDeer

**依赖关系:**
```
Task 1 (decision.py) ──→ Task 2 (morning_update.py) ──→ Task 3 (run_morning.ps1)
Task 4 (autotrade.py) ──→ Task 5 (run_daily.ps1)
```
Task 1 和 Task 4 可并行。Task 2 依赖 Task 1。Task 3 依赖 Task 2。Task 5 依赖 Task 4。

---

### Task 1: 从 ChiefDecisionAgent 提取规则评分静态方法

**Files:**
- Modify: `decision.py`（在文件末尾、`ChiefDecisionAgent`类内新增 `compute_rule_score` 静态方法）
- No tests（纯数学提取，被调用方已验证）

**背景:** 晨盘分析需要复用 `ChiefDecisionAgent.run()` 的规则评分逻辑（z-score标准化 → 加权平均 → 动量修正 → 分歧折扣 → 操作映射 → 凯利仓位），但不需要 LLM 调用路径。当前 `run()` 方法将规则部分和 LLM 部分耦合在一起。

**提取方案:** 新增 `compute_rule_score()` 静态方法，输入为扁平化的评分数据（从快照重建），输出为规则评分结果，不含任何 LLM 调用。

```python
@staticmethod
def compute_rule_score(
    agent_scores: list[dict],    # [{"name": ..., "score": ..., "rating": ...}, ...]
    etf_code: str,
    etf_name: str,
    etf_type: str,
    global_max_pos: float,
    market_state: str = "震荡偏强",
) -> dict:
    """
    纯规则评分（无LLM调用），供晨盘增量分析复用。
    
    参数:
        agent_scores: 各Agent评分列表，来自快照的agents字段
            [{"name": "价值估值智能体", "score": 50, "rating": "中性"},
             {"name": "风险管理智能体", "score": 80, "rating": "看多"}, ...]
        etf_code: ETF代码
        etf_name: ETF名称
        etf_type: ETF类型（宽基/行业/...）
        global_max_pos: 全局仓位上限（macro_score / 100）
        market_state: 市场状态
        
    返回:
        {
            "final_score": float,
            "final_rating": str,
            "operation": str,
            "holding_period": str,
            "position_pct": float,
            "consensus": str,
            "stop_loss_pct": float,
            "take_profit_pct": float,
        }
    """
```

**实现逻辑（从 `ChiefDecisionAgent.run()` 提取规则部分，约50行）:**

- [ ] **Step 1: 在 `ChiefDecisionAgent` 类中新增 `compute_rule_score` 方法签名**

```python
@staticmethod
def compute_rule_score(
    agent_scores: list[dict],
    etf_code: str,
    etf_name: str,
    etf_type: str,
    global_max_pos: float,
    market_state: str = "震荡偏强",
) -> dict:
```

- [ ] **Step 2: 实现评分预处理（RiskAgent反转）**

```python
    adjusted_scores = []
    for a in agent_scores:
        score = a["score"]
        if a["name"] in ChiefDecisionAgent.REVERSE_AGENTS:
            score = 100 - score
        adjusted_scores.append(score)
```

- [ ] **Step 3: 实现共识度计算**

```python
    raw_std = float(np.std(adjusted_scores))
    raw_mean = float(np.mean(adjusted_scores)) if adjusted_scores else 1
    raw_cv = raw_std / max(raw_mean, 1)
    if raw_cv < 0.15:
        consensus = "高度一致"
    elif raw_cv < 0.25:
        consensus = "基本一致"
    elif raw_cv < 0.40:
        consensus = "存在分歧"
    else:
        consensus = "严重分歧"
```

- [ ] **Step 4: 实现 z-score 标准化 → 50 + z*15**

```python
    arr = np.array(adjusted_scores)
    mean, std = np.mean(arr), np.std(arr, ddof=1)
    if std < 1e-6:
        z_scores = [0.0] * len(arr)
    else:
        z_scores = [(s - mean) / std for s in arr]
    normalized_scores = [float(np.clip(50 + z * 15, 0, 100)) for z in z_scores]
```

- [ ] **Step 5: 实现置信度校准 + 动态加权平均**

```python
    # 置信度校准
    calibrated = []
    for i, ns in enumerate(normalized_scores):
        direction = "多" if ns > 55 else "空" if ns < 45 else "中"
        same_dir = 0
        for j, s in enumerate(normalized_scores):
            if j == i:
                continue
            if ("多" if s > 55 else "空" if s < 45 else "中") == direction:
                same_dir += 1
        agreement = same_dir / max(len(normalized_scores) - 1, 1)
        extremity = abs(ns - 50) / 50.0
        confidence = 0.3 + 0.5 * agreement + 0.2 * extremity
        calibrated.append(float(np.clip(confidence, 0.3, 1.0)))
    
    # 加权
    weights = ChiefDecisionAgent.load_agent_weights()
    has_real = any(w != 1.0 for w in weights.values())
    if not has_real:
        weights = ChiefDecisionAgent._load_proxy_weights()
    
    weight_values = []
    for a, cc in zip(agent_scores, calibrated):
        w = weights.get(a["name"], 1.0)
        w *= (0.5 + cc)
        weight_values.append(w)
    
    total_w = sum(weight_values)
    if total_w > 0:
        norm_weights = [w / total_w for w in weight_values]
        weighted_score = sum(n * s for n, s in zip(norm_weights, normalized_scores))
    else:
        weighted_score = float(np.mean(normalized_scores))
```

- [ ] **Step 6: 实现时间序列动量 + 分歧折扣**

```python
    ts_momentum = ChiefDecisionAgent._time_series_momentum_score(etf_code)
    weighted_score += ts_momentum * 0.35
    
    if market_state == "强趋势牛":
        discount = 1.0
    elif market_state == "震荡偏强":
        discount = max(1.0 - max(raw_cv - 0.15, 0) * 0.5, 0.90)
    elif market_state == "震荡偏弱":
        discount = max(1.0 - max(raw_cv - 0.12, 0) * 0.6, 0.85)
    else:
        discount = max(1.0 - max(raw_cv - 0.10, 0) * 0.7, 0.80)
    weighted_score *= discount
    
    final_score = float(np.clip(weighted_score, 0, 100))
    final_rating = ChiefDecisionAgent._score_to_rating(final_score)
```

- [ ] **Step 7: 实现操作映射 + 凯利仓位**

```python
    # 操作映射
    if final_score >= 80:
        operation, holding = ("强烈买入", "短期(1-4周)") if consensus in ("高度一致", "基本一致") else ("买入", "中期(1-3月)")
    elif final_score >= 65:
        operation, holding = "买入", "中期(1-3月)"
    elif final_score >= 50:
        operation, holding = ("长期持有", "长期(6月+)") if etf_type == "宽基" else ("持有", "中期(1-3月)")
    elif final_score >= 35:
        operation, holding = "减持", "短期(1-4周)"
    elif final_score >= 20:
        operation, holding = "卖出", "短期(1-4周)"
    else:
        operation, holding = "强烈卖出", "短期(1-4周)"
    
    if consensus == "严重分歧":
        if operation in ("强烈买入", "买入"):
            operation, holding = "持有", "中期(1-3月)"
        elif operation in ("长期持有",):
            operation, holding = "减持", "短期(1-4周)"
    
    # 凯利仓位
    win_rate = 0.55
    try:
        from review import ReviewManager
        import os
        if os.path.exists(ReviewManager.REVIEW_FILE):
            with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
            if stats.get("total_verifications", 0) > 10:
                win_rate = stats.get("overall_accuracy_pct", 55) / 100
    except Exception:
        pass
    
    market_pos_mult = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}
    pos_mult = market_pos_mult.get(market_state, 1.0)
    adjusted_max_pos = global_max_pos * pos_mult
    kelly_pos = ChiefDecisionAgent._kelly_position(win_rate, 1.5, 1.0, adjusted_max_pos)
    pos_map = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20, "持有": 0.0,
               "减持": 0.0, "卖出": 0.0, "强烈卖出": 0.0}
    baseline_pos = adjusted_max_pos * pos_map.get(operation, 0.1)
    pos_pct = min(kelly_pos, baseline_pos)
```

- [ ] **Step 8: 实现止损止盈 + 返回结果**

```python
    try:
        from data import DataCollectAgent
        df = DataCollectAgent.get_etf_price(etf_code)
        hist_vol = float(df["volatility"].rolling(20).mean().iloc[-1])
        vol_factor = max(hist_vol * 100, 1.0)
        stop_loss_pct = round(-max(vol_factor * 2.0, 3.0), 1)
        take_profit_pct = round(max(vol_factor * 4.0, 6.0), 1)
    except Exception:
        stop_loss_pct = -5.0
        take_profit_pct = 15.0
    
    return {
        "final_score": round(final_score, 1),
        "final_rating": final_rating,
        "operation": operation,
        "holding_period": holding,
        "position_pct": round(pos_pct, 2),
        "consensus": consensus,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
    }
```

- [ ] **Step 9: 验证方法可用**

Run: `python -c "from decision import ChiefDecisionAgent; print(ChiefDecisionAgent.compute_rule_score.__doc__[:50])"`
Expected: 输出 "纯规则评分（无LLM调用），供晨盘增量分析复用。"

- [ ] **Step 10: Commit**

```bash
git add decision.py
git commit -m "refactor: extract compute_rule_score static method from ChiefDecisionAgent"
```

---

### Task 2: 新建 `morning_update.py` — 晨盘增量分析脚本

**Files:**
- Create: `morning_update.py`（~200行）

**依赖:** Task 1 完成（`ChiefDecisionAgent.compute_rule_score` 可用）

- [ ] **Step 1: 创建文件头 + 导入**

```python
"""
ETF 智能投资分析系统 - 晨盘增量更新
8:15 AM 运行：加载昨日快照 + 顶层Agent增量更新 + 规则重评分 → 操盘指导
不调LLM，纯规则评分，轻量快速。
"""
import json
import glob
import os
import numpy as np
from datetime import datetime, date

from config import ETF_POOL, SNAPSHOT_DIR
from data import DataCollectAgent
from agents import (
    MacroAnalystAgent, MonetaryPolicyAgent, PolicyEventAgent, CrossMarketAgent,
)
from decision import ChiefDecisionAgent
from scheduler import MainSchedulerAgent  # 复用 detect_market_state
```

- [ ] **Step 2: 实现快照加载函数**

```python
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), SNAPSHOT_DIR)


def load_latest_snapshot() -> dict | None:
    """加载最新的历史快照，返回 {'date','market_volume_bn','etfs':[...]} 或 None"""
    files = sorted(glob.glob(os.path.join(SNAPSHOT_PATH, "*.json")))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_snapshot_date(data: dict) -> str:
    """从快照提取日期"""
    return data.get("date", data.get("snapshot_date", "unknown"))
```

- [ ] **Step 3: 实现顶层Agent增量更新**

```python
def run_top_level_agents():
    """运行宏观/政策/跨市场Agent，返回新的宏观context和global_max_pos"""
    macro_agent = MacroAnalystAgent()
    monetary_agent = MonetaryPolicyAgent()
    policy_agent = PolicyEventAgent()
    cross_market_agent = CrossMarketAgent()
    
    macro_report = macro_agent.run()
    monetary_report = monetary_agent.run()
    policy_report = policy_agent.run()
    cross_market_report = cross_market_agent.run()
    
    global_max_pos = macro_report.score / 100
    market_state = MainSchedulerAgent().detect_market_state()
    
    macro_summary = macro_report.analysis[:120] if macro_report.analysis else ""
    
    return {
        "macro_score": macro_report.score,
        "global_max_pos": global_max_pos,
        "market_state": market_state,
        "macro_summary": macro_summary,
    }
```

- [ ] **Step 4: 实现逐ETF重评分（复用 compute_rule_score）**

```python
def re_score_etfs(snapshot_etfs: list, top: dict) -> list[dict]:
    """对快照中的每只ETF应用新的宏观背景重新评分"""
    results = []
    for etf in snapshot_etfs:
        agent_scores = etf.get("agents", [])
        if not agent_scores:
            continue
        
        result = ChiefDecisionAgent.compute_rule_score(
            agent_scores=agent_scores,
            etf_code=etf["code"],
            etf_name=etf["name"],
            etf_type=etf.get("type", ""),
            global_max_pos=top["global_max_pos"],
            market_state=top["market_state"],
        )
        
        # 对比新旧结果
        old_op = etf.get("operation", "")
        old_score = etf.get("final_score", 0)
        new_op = result["operation"]
        new_score = result["final_score"]
        
        changed = old_op != new_op
        score_delta = round(new_score - old_score, 1)
        
        results.append({
            "code": etf["code"],
            "name": etf["name"],
            "type": etf.get("type", ""),
            "old_operation": old_op,
            "new_operation": new_op,
            "old_score": old_score,
            "new_score": new_score,
            "score_delta": score_delta,
            "position_pct": result["position_pct"],
            "consensus": result["consensus"],
            "changed": changed,
        })
    
    return results
```

- [ ] **Step 5: 实现组合约束（相关性 + 总仓位）**

```python
def apply_portfolio_constraints(results: list[dict]) -> list[dict]:
    """应用总仓位归一化和相关性约束"""
    # 总仓位 ≤ 100%
    total_pos = sum(r["position_pct"] for r in results)
    if total_pos > 1.0:
        scale = 1.0 / total_pos
        for r in results:
            r["position_pct"] = round(r["position_pct"] * scale, 4)
    
    # 相关性约束（复用 scheduler 的逻辑）
    # 对有仓位的ETF做相关性约束
    actionable = [r for r in results if r["position_pct"] > 0]
    if len(actionable) >= 2:
        price_data = {}
        for r in actionable:
            try:
                df = DataCollectAgent.get_etf_price(r["code"])
                if len(df) >= 60:
                    price_data[r["code"]] = df["close"].pct_change().dropna().tail(60).values
            except Exception:
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
```

- [ ] **Step 6: 实现操盘指导文本格式化**

```python
def format_guidance(results: list[dict], top: dict, snapshot_date: str) -> str:
    """格式化为推送用的操盘指导文本"""
    today_str = datetime.now().strftime("%m月%d日")
    lines = []
    lines.append(f"🏆 今日操盘指导 · {today_str}")
    lines.append("")
    lines.append(f"【市场状态】{top['market_state']} | 仓位上限 {top['global_max_pos']*100:.0f}%")
    if top["macro_summary"]:
        lines.append(f"宏观: {top['macro_summary']}")
    lines.append("")
    
    # 强烈买入
    strong_buys = [r for r in results if r["new_operation"] == "强烈买入"]
    if strong_buys:
        lines.append(f"🔥 强烈买入 ({len(strong_buys)})")
        for r in strong_buys:
            lines.append(f"  {r['code']} {r['name']}  仓位{r['position_pct']*100:.0f}%")
        lines.append("")
    
    # 买入
    buys = [r for r in results if r["new_operation"] == "买入"]
    if buys:
        lines.append(f"📈 买入 ({len(buys)})")
        for r in buys:
            lines.append(f"  {r['code']} {r['name']}  仓位{r['position_pct']*100:.0f}%")
        lines.append("")
    
    # 风险信号 (减持/卖出/强烈卖出)
    risks = [r for r in results if r["new_operation"] in ("减持", "卖出", "强烈卖出")]
    if risks:
        lines.append("⚠️ 风险信号")
        for r in risks:
            lines.append(f"  {r['code']} {r['name']} → {r['new_operation']} (评分{r['new_score']:.0f})")
        lines.append("")
    
    # 变化提示
    changes = [r for r in results if r["changed"]]
    if changes:
        lines.append("📊 变化提示")
        for r in changes[:5]:  # 最多显示5条
            delta_str = f"{r['score_delta']:+.1f}"
            lines.append(f"  {r['code']}: {r['old_operation']}→{r['new_operation']} (评分{r['old_score']:.0f}→{r['new_score']:.0f}, {delta_str})")
        lines.append("")
    
    # 总仓位
    total_pos = sum(r["position_pct"] for r in results)
    lines.append(f"【建议总仓位】{total_pos*100:.1f}%")
    lines.append(f"数据基于{snapshot_date}快照 + 今日增量更新")
    
    return "\n".join(lines)
```

- [ ] **Step 7: 实现 main 函数**

```python
def main():
    snapshot = load_latest_snapshot()
    if snapshot is None:
        msg = "尚无历史快照数据，请先运行午后全量分析 (python etf-agent.py)"
        print(msg)
        return
    
    snap_date = get_snapshot_date(snapshot)
    print(f"📂 加载快照: {snap_date}, {len(snapshot.get('etfs', []))} 只ETF")
    
    # 检查日期：提醒非昨日数据
    try:
        snap_dt = datetime.strptime(snap_date, "%Y%m%d")
        days_old = (datetime.now() - snap_dt).days
        if days_old > 1:
            print(f"⚠️ 警告: 快照日期为{snap_date}({days_old}天前)，数据可能陈旧")
    except Exception:
        pass
    
    print(f"\n🔄 增量更新: 宏观/政策/跨市场Agent...")
    top = run_top_level_agents()
    print(f"  ✅ 宏观: {top['macro_score']:.0f}分 | 市场状态: {top['market_state']} | 仓位上限: {top['global_max_pos']*100:.0f}%")
    
    print(f"\n🔄 规则重评分: {len(snapshot['etfs'])} 只ETF...")
    results = re_score_etfs(snapshot["etfs"], top)
    
    print(f"\n🔄 组合约束...")
    results = apply_portfolio_constraints(results)
    
    changed_count = sum(1 for r in results if r["changed"])
    print(f"  ✅ 变化: {changed_count} 只ETF操作建议改变")
    
    guidance = format_guidance(results, top, snap_date)
    print("\n" + "="*60)
    print(guidance)
    print("="*60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: 测试运行**

Run: `python morning_update.py`
Expected: 正常输出操盘指导文本

- [ ] **Step 9: Commit**

```bash
git add morning_update.py
git commit -m "feat: 晨盘增量分析脚本 morning_update.py"
```

---

### Task 3: 新建 `run_morning.ps1` — 晨盘编排推送脚本

**Files:**
- Create: `run_morning.ps1`（~80行）

**依赖:** Task 2 完成

- [ ] **Step 1: 创建 `run_morning.ps1`**

```powershell
<#
.SYNOPSIS
  ETF晨盘增量分析推送脚本
.DESCRIPTION
  运行 morning_update.py 生成操盘指导并通过 PushDeer 推送
.NOTES
  计划任务配置: 每天 08:15 运行
  程序: powershell
  参数: -ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_morning.ps1"
#>

param()

$ProjectDir   = "D:\project\ETF投资策略"
$LogDir       = Join-Path $ProjectDir "logs"
$DateStr      = Get-Date -Format "yyyyMMdd"
$LogFile      = Join-Path $LogDir "${DateStr}_morning.log"
$EnvFile      = Join-Path $ProjectDir ".env"
$PushDeerKey  = if (Test-Path $EnvFile) {
    $envContent = Get-Content $EnvFile -Encoding utf8
    $line = $envContent | Where-Object { $_ -match "^PUSHDEER_KEY=" } | Select-Object -First 1
    if ($line) { $line -replace "^PUSHDEER_KEY=", "" } else { "" }
} else { "" }
$PushDeerUrl  = "https://api2.pushdeer.com/message/push"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append -Encoding utf8
    Write-Host "$timestamp - $Message"
}

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Set-Location $ProjectDir
Write-Log "=== ETF Morning Analysis Start ==="

try {
    # 运行晨盘分析，捕获输出
    $output = python -X utf8 morning_update.py 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -ne 0) {
        throw "Python exited with code $exitCode"
    }
    
    Write-Log "morning_update.py completed successfully"
    
    # 从输出中提取摘要部分（从 🏆 到文件尾）
    $guidanceLines = $output | Out-String
    $startIndex = $guidanceLines.IndexOf("🏆")
    if ($startIndex -ge 0) {
        $guidanceText = $guidanceLines.Substring($startIndex).Trim()
    } else {
        # fallback: 取最后50行
        $guidanceText = $output | Select-Object -Last 50 | Out-String
    }
    
    # 推送
    $encoded = [Uri]::EscapeDataString($guidanceText)
    $pushUrl = "${PushDeerUrl}?pushkey=${PushDeerKey}&text=${encoded}"
    try {
        $resp = Invoke-RestMethod -Uri $pushUrl -Method Get
        Write-Log "PushDeer notification sent (success)"
        Write-Host "推送成功"
    } catch {
        Write-Log "PushDeer notification failed: $_"
        Write-Host "推送失败: $_"
    }
} catch {
    Write-Log "ERROR: $_"
    $errMsg = "ETF晨盘分析失败: $_"
    $encodedErr = [Uri]::EscapeDataString($errMsg)
    try {
        Invoke-RestMethod -Uri "${PushDeerUrl}?pushkey=${PushDeerKey}&text=${encodedErr}" -Method Get | Out-Null
        Write-Log "PushDeer error notification sent"
    } catch {
        Write-Log "PushDeer notification also failed: $_"
    }
}

Write-Log "=== ETF Morning Analysis End ==="
```

- [ ] **Step 2: 测试运行**

Run: `.\run_morning.ps1`
Expected: 执行 morning_update.py 并推送

- [ ] **Step 3: Commit**

```bash
git add run_morning.ps1
git commit -m "feat: 晨盘编排推送脚本 run_morning.ps1"
```

---

### Task 4: 修改 `autotrade.py` — 增加模拟盘日盈亏字段

**Files:**
- Modify: `autotrade.py`（`_log_performance` 函数内，约15行增量）

- [ ] **Step 1: 在 `_log_performance` 函数中增加 daily P&L 计算**

修改 `_log_performance` 函数（`autotrade.py:479-524`），在夏普计算之后、`last_updated` 之前插入日盈亏计算：

```python
    # 日盈亏计算（从 trade_log 最近两天差值）
    try:
        if TRADE_LOG.exists():
            with open(TRADE_LOG, "r", encoding="utf-8") as f:
                snaps = json.load(f)
            if len(snaps) >= 2:
                prev_total = snaps[-2].get("total", total)
                daily_pnl = total - prev_total
                daily_pnl_pct = round((daily_pnl / prev_total) * 100, 2) if prev_total > 0 else 0
                perf["daily_pnl"] = round(daily_pnl, 2)
                perf["daily_pnl_pct"] = daily_pnl_pct
            elif len(snaps) == 1:
                # 第一天，无前日对比
                perf["daily_pnl"] = round(total - perf["initial_capital"], 2)
                perf["daily_pnl_pct"] = round(perf["total_return_pct"], 2)
            cumulative_pnl = total - perf["initial_capital"]
            perf["cumulative_pnl"] = round(cumulative_pnl, 2)
    except Exception:
        pass

    perf["last_updated"] = date
```

**插入位置:** 在 `perf["sharpe_ratio"] = ...` 之后（第518行）、`perf["last_updated"] = date`（第522行）之前。

- [ ] **Step 2: 验证写入**

Run: `python -c "from autotrade import show_performance; show_performance()"`
Expected: 正常运行，不报错（性能数据不变）

- [ ] **Step 3: Commit**

```bash
git add autotrade.py
git commit -m "feat: performance.json 增加 daily_pnl/cumulative_pnl 字段"
```

---

### Task 5: 修改 `run_daily.ps1` — 推送增加模拟盘当日盈亏

**Files:**
- Modify: `run_daily.ps1`（在推送 body 构建部分增加模拟盘盈亏段落，约15行增量）

**依赖:** Task 4 完成（performance.json 已有 daily_pnl 字段）

- [ ] **Step 1: 修改 `run_daily.ps1` 中推送 body 的构建部分**

在现有推送 body（第74-82行）的 `【策略表现】` 板块与 `【调仓明细】` 板块之间插入模拟盘盈亏段落：

```powershell
# 读取 performance.json 获取当日盈亏
$PerfFile = Join-Path $ProjectDir "data" "performance.json"
$dailyPnlBlock = ""
if (Test-Path $PerfFile) {
    try {
        $perfContent = Get-Content $PerfFile -Encoding utf8 | ConvertFrom-Json
        $pnl = $perfContent.daily_pnl
        $pnlPct = $perfContent.daily_pnl_pct
        $cumPnl = $perfContent.cumulative_pnl
        $marketVal = $perfContent.market_value
        $cashVal = $perfContent.cash
        if ($pnl -ne $null) {
            $pnlSign = if ($pnl -ge 0) { "+" } else { "" }
            $dailyPnlBlock = @"

【模拟盘当日盈亏】
当日盈亏: ${pnlSign}${pnl} (${pnlSign}${pnlPct}%)
累计盈亏: ${cumPnl}
持仓市值: ${marketVal}
现金余额: ${cashVal}
"@
        }
    } catch {
        Write-Log "Failed to read performance.json: $_"
    }
}
```

然后将 `$dailyPnlBlock` 嵌入到 `$body` 中，放在 `【策略表现】` 和 `【调仓明细】` 之间。

**具体修改:**
找到原有 body 构建代码（约第74行）：
```powershell
    $body = @"
ETF分析完成

【策略表现】
$($perfLines -join "`n")

【调仓明细】
$($tradeLines -join "`n")
"@
```

改为：
```powershell
    $body = @"
ETF分析完成

【策略表现】
$($perfLines -join "`n")
$dailyPnlBlock

【调仓明细】
$($tradeLines -join "`n")
"@
```

- [ ] **Step 2: 验证推送格式**

Run: `.\run_daily.ps1`（确保在15:00后，或临时跳过时间检查）
Expected: 推送内容包含 【模拟盘当日盈亏】段落

- [ ] **Step 3: Commit**

```bash
git add run_daily.ps1
git commit -m "feat: 午后推送增加模拟盘当日盈亏"
```

---

## 自检清单

- [ ] **Spec 覆盖:** 所有 spec 要求的需求都有对应的 task 实现
- [ ] **无占位符:** 每个 step 都有完整代码或命令
- [ ] **类型一致性:** compute_rule_score 的输入输出类型在 Task 1 和 Task 2 间一致
