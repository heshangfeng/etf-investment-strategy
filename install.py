import subprocess
import sys

def install_package(package):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        package, "-q", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
    ])

# 基础库 + 爬虫 + NLP + 大模型 依赖包
install_package("akshare")
install_package("pandas")
install_package("numpy")
install_package("requests")
install_package("beautifulsoup4")
install_package("jieba")
install_package("snownlp")
install_package("openai")
install_package("python-dotenv")
install_package("scipy")
install_package("streamlit")

# 新增：投研逻辑增强库
install_package("pyportfolioopt")    # 组合优化（均值-方差/Black-Litterman/HRP）
install_package("plotly")            # 交互图表（K线/雷达/热力）
install_package("instructor")        # LLM结构化输出（替代手写JSON解析）
install_package("scikit-learn")      # 置信度校准/标准化/集成方法
install_package("tushare")           # 金融数据备份源
install_package("structlog")         # 结构化日志（替代print）
install_package("tenacity")          # API调用重试

print("✅ 全部依赖安装完成！")