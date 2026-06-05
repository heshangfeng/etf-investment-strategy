<#
.SYNOPSIS
  ETF每日投研分析自动化脚本（委托 etf-agent.py，推送由 Python 内部处理）
.DESCRIPTION
  运行 etf-agent.py 完成全量分析 + PushDeer 推送，不再在 PS1 中处理推送逻辑
.NOTES
  计划任务配置: 每天 15:30 运行
  程序: powershell
  参数: -ExecutionPolicy Bypass -File "D:\project\ETF投资策略\run_daily.ps1"
#>

param()
$ProjectDir = "D:\project\ETF投资策略"
if (-not (Test-Path (Join-Path $ProjectDir "logs"))) {
    New-Item -ItemType Directory -Path (Join-Path $ProjectDir "logs") -Force | Out-Null
}
Set-Location $ProjectDir
$output = python -X utf8 etf-agent.py 2>&1
$output | Out-File -FilePath (Join-Path $ProjectDir "logs\$(Get-Date -Format yyyyMMdd)_run.log") -Append -Encoding utf8
Write-Host $output
