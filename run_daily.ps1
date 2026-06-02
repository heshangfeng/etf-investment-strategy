<#
.SYNOPSIS
  ETF每日投研分析自动化脚本
.DESCRIPTION
  运行 etf-agent.py 生成投研报告并通过 PushDeer 发送通知
.NOTES
  计划任务配置: 每天 15:30 运行
  程序: powershell
  参数: -ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_daily.ps1"
#>

param()

$ProjectDir   = "D:\project\ETF投资策略"
$LogDir       = Join-Path $ProjectDir "logs"
$DateStr      = Get-Date -Format "yyyyMMdd"
$LogFile      = Join-Path $LogDir "${DateStr}_run.log"
$PushDeerKey  = "PUSHDEER_KEY_REMOVED_PLEASE_SET_NEW_KEY"
$PushDeerUrl  = "https://api2.pushdeer.com/message/push"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append -Encoding utf8
    Write-Host "$timestamp - $Message"
}

# ---- init ----
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Set-Location $ProjectDir
Write-Log "=== ETF Daily Analysis Start ==="

# ---- time check (>15:00) ----
$now = Get-Date
$marketClose = Get-Date -Hour 15 -Minute 0 -Second 0
if ($now -lt $marketClose) {
    Write-Log "WARNING: Market not closed yet ($($now.ToString('HH:mm'))). Before 15:00."
    $confirm = Read-Host "Market still open. Continue? (y/N)"
    if ($confirm -ne "y") {
        Write-Log "Aborted by user"
        exit 0
    }
}

# ---- run python ----
try {
    $output = python -X utf8 etf-agent.py 2>&1
    $exitCode = $LASTEXITCODE
    $output | Out-File -FilePath $LogFile -Append -Encoding utf8

    if ($exitCode -ne 0) {
        throw "Python exited with code $exitCode"
    }

    Write-Log "Python script completed successfully"

    # ---- find latest txt report ----
    $reportFile = Get-ChildItem -Path $ProjectDir -Filter "ETF_多智能体投研报告_${DateStr}.txt" -ErrorAction SilentlyContinue `
        | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if (-not $reportFile) {
        $reportFile = Get-ChildItem -Path $ProjectDir -Filter "ETF_多智能体投研报告_*.txt" -ErrorAction SilentlyContinue `
            | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }

    if ($reportFile) {
        $lines = Get-Content -Path $reportFile.FullName -Encoding utf8
        $titleLine   = $lines | Select-Object -Index 1   # report title with date
        $summaryLine = $lines | Select-Object -Index 5   # 【投研摘要看板】

        $etfLines = $lines | Where-Object { $_ -match "^\s{2}\d{6}" } | Select-Object -First 5

        $body = @"
$titleLine

【操作建议摘要】
$($etfLines -join "`n")

详细报告: $($reportFile.Name)
"@

        $encoded = [Uri]::EscapeDataString($body)
        $pushUrl = "${PushDeerUrl}?pushkey=${PushDeerKey}&text=${encoded}"
        $resp = Invoke-RestMethod -Uri $pushUrl -Method Get
        Write-Log "PushDeer notification sent (success)"
        Write-Log "PushDeer response: $($resp | Out-String)"
    } else {
        Write-Log "No report txt file found"
    }
} catch {
    Write-Log "ERROR: $_"
    $errMsg = "ETF分析运行失败: $_ | 详见日志文件 ${DateStr}_run.log"
    $encodedErr = [Uri]::EscapeDataString($errMsg)
    $pushUrl = "${PushDeerUrl}?pushkey=${PushDeerKey}&text=${encodedErr}"
    try {
        Invoke-RestMethod -Uri $pushUrl -Method Get | Out-Null
        Write-Log "PushDeer error notification sent"
    } catch {
        Write-Log "PushDeer notification also failed: $_"
    }
}

Write-Log "=== ETF Daily Analysis End ==="

<#
Windows Task Scheduler 配置说明
───────────────────────────────────────
1. 打开"任务计划程序" → 创建基本任务
2. 名称: ETF Daily Analysis
3. 触发器: 每天, 15:30
4. 操作: 启动程序
   程序: powershell
   参数: -ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_daily.ps1"
   起始于: D:\project\ETF投资策略
5. 完成

注意:
- 如果系统区域设置为中文，PowerShell 默认编码可能不是 UTF-8，
  建议在计划任务中勾选"使用最高权限运行"
- 首次运行前请将脚本中的 PUSHDEER_KEY 替换为真实的 PushDeer Key
#>
