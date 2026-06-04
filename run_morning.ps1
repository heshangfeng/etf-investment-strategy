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
