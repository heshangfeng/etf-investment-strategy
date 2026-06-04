"""
ETF 智能投资分析系统 - 入口文件
"""
# 自动更新项目目录结构文档（静默，失败不影响分析）
try:
    from scripts.generate_structure import generate_markdown, update_run_command_md
    _md = generate_markdown()
    update_run_command_md(_md)
except Exception:
    pass

from core.scheduler import main

if __name__ == "__main__":
    main()

