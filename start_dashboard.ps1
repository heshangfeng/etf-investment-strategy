# ETF 智能投研看板 - 启动脚本（带自动重启）
# 双击运行即可，看板崩溃后自动重启

$env:STREAMLIT_EMAIL = ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ETF 智能投研看板 - 启动中..."         -ForegroundColor Cyan
Write-Host "  地址: http://localhost:8501"           -ForegroundColor Cyan
Write-Host "  按 Ctrl+C 停止"                       -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

while ($true) {
    try {
        streamlit run dashboard.py --server.headless true
    }
    catch {
        Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] 看板异常退出: $_" -ForegroundColor Red
    }

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 看板已退出，5 秒后重启... (按 Ctrl+C 终止)" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
