"""
ETF 智能投资分析系统 - 全局配置与缓存
"""
import os
import sys
import warnings
from dotenv import load_dotenv
from infra.cache import PersistentCache
from infra.logger import get_logger

logger = get_logger("config")

warnings.filterwarnings("ignore")
load_dotenv()

# ====================== 【全局核心配置区 - 可直接修改】 ======================
# 1. 综合评分权重
WEIGHT = {
    "value": 0.12,
    "boom": 0.20,
    "tech": 0.20,
    "fund": 0.16,
    "risk": 0.16,
    "opinion": 0.16
}

# 2. 风控 & 舆情阈值
PREMIUM_RISK_THRESHOLD = 0.015
VOL_RISK_THRESHOLD = 0.03
LIQ_THRESHOLD = 5000
OPINION_WARN_THRESHOLD = 30    # 利空告警线
TREND_DAY_COUNT = 7           # 舆情趋势统计天数

# 3. 并行 & 网络配置
MAIN_WORKERS = 6
AGENT_WORKERS = 8
REQUEST_DELAY = 0.1
FEE_RATE = 0.0003             # ETF 交易费率（万三）
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 4. 大模型配置（从 .env 读取）
if not os.getenv("DEEPSEEK_API_KEY"):
    logger.warning(".env 未找到或 DEEPSEEK_API_KEY 未设置，LLM 模式将不可用")
    logger.warning("请创建 .env 文件: echo DEEPSEEK_API_KEY=sk-xxx > .env")

# ── 深度模型（复杂推理：首席决策、辩论、宏观、政策） ──
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_MAX_TOKENS = 2048                 # 深度模型 token 上限
LLM_TEMPERATURE = 0.3
LLM_ENABLED = True                    # 总开关：True=LLM多智能体模式

# ── 快速模型（轻量分析：价值/技术/情绪/资金/风控/行业/零售/跨市场） ──
QUICK_LLM_API_KEY = os.getenv("QUICK_LLM_API_KEY", LLM_API_KEY)
QUICK_LLM_BASE_URL = os.getenv("QUICK_LLM_BASE_URL", LLM_BASE_URL)
QUICK_LLM_MODEL = os.getenv("QUICK_LLM_MODEL", "deepseek-chat")
QUICK_LLM_MAX_TOKENS = int(os.getenv("QUICK_LLM_MAX_TOKENS", "1024"))
QUICK_LLM_TEMPERATURE = 0.4

# LLM 调用统计
LLM_CALL_COUNT = {"deep": 0, "quick": 0, "total": 0}

# 6. 多智能体辩论配置
DEBATE_ENABLED = True
DEBATE_ROUNDS = 2
DISAGREEMENT_SCORE_THRESHOLD = 18    # 评分差异≥此值触发辩论
DISAGREEMENT_RATING_GAP = 2          # 评级级差≥此值触发辩论

RATING_ORDER = ["强烈看空", "看空", "中性", "看多", "强烈看多"]

# ====================== 【项目路径配置】 ======================
OUTPUT_DIR = "output"               # 报告文件输出目录
SNAPSHOT_DIR = "data/snapshots"     # 每日快照目录
REVIEW_DIR = "data/review"          # 复盘数据目录
CACHE_DB_PATH = "data/cache.db"     # SQLite 缓存数据库

# ====================== 【ETF标的池】 ======================
ETF_POOL = [
    # ── 宽基 (8只) ──
    {"code": "510050", "name": "上证50ETF",     "type": "宽基", "index_code": "000016"},
    {"code": "510300", "name": "沪深300ETF",    "type": "宽基", "index_code": "000300"},
    {"code": "159338", "name": "中证A500ETF",   "type": "宽基", "index_code": "000510"},
    {"code": "510500", "name": "中证500ETF",    "type": "宽基", "index_code": "000905"},
    {"code": "512100", "name": "中证1000ETF",   "type": "宽基", "index_code": "000852"},
    {"code": "563300", "name": "中证2000ETF",   "type": "宽基", "index_code": "932000"},
    {"code": "159915", "name": "创业板ETF",     "type": "宽基", "index_code": "399006"},
    {"code": "588000", "name": "科创50ETF",     "type": "宽基", "index_code": "000688"},
    # ── 行业 (18只) ──
    {"code": "512000", "name": "券商ETF",       "type": "行业", "index_code": "801780"},
    {"code": "512800", "name": "银行ETF",       "type": "行业", "index_code": "801780"},
    {"code": "512400", "name": "有色金属ETF",   "type": "行业", "index_code": "801050"},
    {"code": "515220", "name": "煤炭ETF",       "type": "行业", "index_code": "801950"},
    {"code": "512200", "name": "房地产ETF",     "type": "行业", "index_code": "801180"},
    {"code": "159611", "name": "电力ETF",       "type": "行业", "index_code": "000993"},
    {"code": "515080", "name": "基建ETF",       "type": "行业", "index_code": "801730"},
    {"code": "512760", "name": "半导体ETF",     "type": "行业", "index_code": "BK1036"},
    {"code": "512480", "name": "半导体设备ETF", "type": "行业", "index_code": "BK1036"},
    {"code": "512690", "name": "酒ETF",         "type": "行业", "index_code": "801120"},
    {"code": "159928", "name": "消费ETF",       "type": "行业", "index_code": "801110"},
    {"code": "159858", "name": "医药ETF",       "type": "行业", "index_code": "801150"},
    {"code": "512290", "name": "生物医药ETF",   "type": "行业", "index_code": "399441"},
    {"code": "515790", "name": "光伏ETF",       "type": "行业", "index_code": "BK0448"},
    {"code": "159806", "name": "风电ETF",       "type": "行业", "index_code": "BK1032"},
    {"code": "159875", "name": "新能源车ETF",   "type": "行业", "index_code": "BK0493"},
    {"code": "516150", "name": "汽车ETF",       "type": "行业", "index_code": "801020"},
    {"code": "516670", "name": "稀土ETF",       "type": "行业", "index_code": "BK0546"},
    # ── 主题 (14只) ──
    {"code": "512660", "name": "军工ETF",       "type": "主题", "index_code": "801740"},
    {"code": "159819", "name": "人工智能ETF",   "type": "主题", "index_code": "BK1073"},
    {"code": "515070", "name": "大数据ETF",     "type": "主题", "index_code": "BK1049"},
    {"code": "516950", "name": "机器人ETF",     "type": "主题", "index_code": "BK0964"},
    {"code": "512980", "name": "传媒ETF",       "type": "主题", "index_code": "801760"},
    {"code": "159825", "name": "农业ETF",       "type": "主题", "index_code": "801010"},
    {"code": "513090", "name": "恒生科技ETF",   "type": "主题", "index_code": "HSTECH"},
    {"code": "513130", "name": "港股互联网ETF", "type": "主题", "index_code": "H11136"},
    {"code": "518880", "name": "黄金ETF",       "type": "主题", "index_code": "000016"},
    {"code": "510880", "name": "红利ETF",       "type": "主题", "index_code": "000015"},
    {"code": "513100", "name": "纳指ETF",       "type": "主题", "index_code": "NDX"},
    {"code": "513050", "name": "中概互联ETF",   "type": "主题", "index_code": "H11136"},
    {"code": "516220", "name": "元宇宙ETF",     "type": "主题", "index_code": "BK0992"},
    {"code": "515680", "name": "数字货币ETF",   "type": "主题", "index_code": "BK0887"},
]

# 5. 外部财经新闻API
FIN_API_KEY = os.getenv("FIN_API_KEY", "")
FIN_API_URL = os.getenv("FIN_API_URL", "")

# ====================== 【全局缓存 - 带TTL】 ======================
CACHE_ETF_PRICE = PersistentCache("etf_price", default_ttl=3600)
CACHE_INDEX_VAL = PersistentCache("index_val", default_ttl=7200)
CACHE_ETF_PREMIUM = PersistentCache("etf_premium", default_ttl=3600)
CACHE_NORTH_CAP = PersistentCache("north_cap", default_ttl=3600)
CACHE_MARKET_VOL = PersistentCache("market_vol", default_ttl=1800)
CACHE_OPINION = PersistentCache("opinion", default_ttl=3600)
CACHE_OPINION_HIST = PersistentCache("opinion_hist", default_ttl=86400)
CACHE_IVIX = PersistentCache("ivix", default_ttl=3600)

