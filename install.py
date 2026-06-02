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

print("✅ 全部依赖安装完成！")