import akshare as ak
import pandas as pd
import numpy as np
from math import comb, log, exp
import warnings
import time
import re
import json
import os
import glob
import jieba
import requests
from bs4 import BeautifulSoup
from snownlp import SnowNLP
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

warnings.filterwarnings("ignore")

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
TREND_DAY_COUNT = 3           # 舆情趋势统计天数

# 3. 并行 & 网络配置
MAIN_WORKERS = 6
AGENT_WORKERS = 8
REQUEST_DELAY = 0.1
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 4. 大模型配置（替换为你的API信息）
LLM_API_KEY = "sk-LEAKED_KEY_REMOVED_PLEASE_SET_NEW_KEY"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"           # DeepSeek高性价比模型
LLM_MAX_TOKENS = 2048                 # 增加到2048以支持详细分析
LLM_TEMPERATURE = 0.3
LLM_ENABLED = True                    # 总开关：True=LLM多智能体模式

# 5. 专业财经API配置（预留接口，可替换商用API）
FIN_API_KEY = "your-fin-api-key"
FIN_API_URL = "https://api.finance.example.com/news"

# 6. 多智能体辩论配置
DEBATE_ENABLED = True
DEBATE_ROUNDS = 2
DISAGREEMENT_SCORE_THRESHOLD = 18    # 评分差异≥此值触发辩论
DISAGREEMENT_RATING_GAP = 2          # 评级级差≥此值触发辩论

RATING_ORDER = ["强烈看空", "看空", "中性", "看多", "强烈看多"]

from dataclasses import dataclass, field

# ====================== 【数据类定义】 ======================

@dataclass
class AgentReport:
    """单个智能体的分析报告"""
    agent_name: str
    etf_code: str
    etf_name: str
    rating: str                # 强烈看多 / 看多 / 中性 / 看空 / 强烈看空
    score: float               # 0-100
    analysis: str              # LLM分析文本
    key_factors: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.5
    data_summary: dict = field(default_factory=dict)
    source: str = "llm"        # "llm" 或 "rule_fallback"


@dataclass
class FinalResearchReport:
    """单只ETF的完整投研报告"""
    etf_info: dict
    macro_context: str = ""
    agent_reports: list[AgentReport] = field(default_factory=list)
    debates: list = field(default_factory=list)
    final_rating: str = "未评级"
    position_suggestion: str = "观望"
    suggested_position_pct: float = 0.0
    final_score: float = 0.0       # 首席综合评分
    core_logic: str = ""
    risk_summary: str = ""
    consensus_level: str = "未知"
    factor_contributions: dict = field(default_factory=dict)
    operation: str = "观望"        # 操作建议：强烈买入/买入/长期持有/持有/减持/卖出/强烈卖出
    holding_period: str = ""      # 持有周期：短期(1-4周)/中期(1-3月)/长期(6月+)
    stop_loss_pct: float = 0.0    # 建议止损位（负值，如-5.0表示跌5%止损）
    take_profit_pct: float = 0.0  # 建议止盈位（正值，如+15.0表示涨15%止盈）


# ====================== 【分层关键词库 - 行业专属】 ======================
# 通用多空词库
BASE_POS_KEYWORDS = {"利好", "政策支持", "增长", "上涨", "回暖", "突破", "资金流入", "景气", "盈利"}
BASE_NEG_KEYWORDS = {"利空", "下跌", "亏损", "风险", "回调", "减持", "监管收紧", "下滑", "暴雷"}

# 行业专属正向词
INDUSTRY_POS = {
    "半导体": {"国产替代", "芯片涨价", "产能扩张", "技术突破", "订单饱满"},
    "医药": {"集采缓和", "新药获批", "医保扩容", "出海提速"},
    "券商": {"成交量放大", "降准降息", "财富管理", "投行业务增长"},
    "光伏": {"装机大增", "组件涨价", "海外订单", "政策补贴"},
    "新能源车": {"销量创新高", "电池技术升级", "出口增长"},
    "银行": {"息差企稳", "不良下降", "信贷放量"},
    "军工": {"订单落地", "国防预算提升", "产业链景气"}
}

# 行业专属负向词
INDUSTRY_NEG = {
    "半导体": {"库存高企", "价格战", "产能过剩"},
    "医药": {"集采扩面", "药品降价", "研发不及预期"},
    "券商": {"成交额低迷", "佣金下滑"},
    "光伏": {"产能过剩", "价格下跌", "贸易壁垒"},
    "新能源车": {"补贴退坡", "销量下滑", "价格战"}
}

# ====================== 【全局缓存 - 带TTL】 ======================
CACHE_ETF_PRICE = {}
CACHE_INDEX_VAL = {}
CACHE_ETF_PREMIUM = {}
CACHE_NORTH_CAP = {}
CACHE_MARKET_VOL: float = 0
CACHE_OPINION = {}          # 当日舆情缓存
CACHE_OPINION_HIST = {}     # 历史舆情时序缓存（趋势用）
CACHE_TIMESTAMPS = {}       # key→最后更新时间, 用于TTL检查
CACHE_IVIX = {}            # 隐含波动率VIX缓存 {etf_code: value}

def _cache_check(cache_key: str, max_age_seconds: int = 3600) -> bool:
    """检查缓存是否过期。过期返回False（需刷新），有效返回True。"""
    ts = CACHE_TIMESTAMPS.get(cache_key, 0)
    return time.time() - ts < max_age_seconds

def _cache_set(cache_key: str):
    """设置缓存时间戳。"""
    CACHE_TIMESTAMPS[cache_key] = time.time()

# ====================== 【1. 基础数据采集智能体】 ======================
class DataCollectAgent:
    @staticmethod
    def get_market_total_volume() -> float:
        global CACHE_MARKET_VOL
        if CACHE_MARKET_VOL > 0 and _cache_check("market_vol", 1800):
            return CACHE_MARKET_VOL
        try:
            # SSE成交金额(亿)
            sse = ak.stock_sse_deal_daily()
            sse_vol = float(sse.loc[sse["单日情况"] == "成交金额", "股票"].values[0])
            # SZSE成交金额(元→亿)
            szse = ak.stock_szse_summary()
            szse_vol = float(szse.loc[szse["证券类别"] == "股票", "成交金额"].values[0]) / 1e8
            vol = round(sse_vol + szse_vol, 1)
        except Exception:
            if CACHE_MARKET_VOL > 0:
                return CACHE_MARKET_VOL
            vol = 7000.0  # fallback中性值
        CACHE_MARKET_VOL = vol
        _cache_set("market_vol")
        return vol

    @staticmethod
    def _etf_code_with_prefix(code: str) -> str:
        """ETF代码转新浪格式：510300 → sh510300, 159915 → sz159915"""
        if code.startswith(("51", "58")):
            return f"sh{code}"
        return f"sz{code}"

    @staticmethod
    def get_etf_price(etf_code: str) -> pd.DataFrame:
        time.sleep(REQUEST_DELAY)
        cache_key = f"price_{etf_code}"
        if etf_code in CACHE_ETF_PRICE and _cache_check(cache_key, 3600):
            return CACHE_ETF_PRICE[etf_code]
        df = ak.fund_etf_hist_sina(symbol=DataCollectAgent._etf_code_with_prefix(etf_code))
        df = df.sort_values("date").reset_index(drop=True)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["volatility"] = df["close"].pct_change().abs()
        CACHE_ETF_PRICE[etf_code] = df
        _cache_set(cache_key)
        return df

    @staticmethod
    def get_index_val(index_code: str) -> dict:
        time.sleep(REQUEST_DELAY)
        cache_key = f"idx_{index_code}"
        if index_code in CACHE_INDEX_VAL and _cache_check(cache_key, 7200):
            return CACHE_INDEX_VAL[index_code]
        try:
            df = ak.stock_zh_index_valuation(symbol=index_code)
            pe = df["PE(TTM)"].iloc[-1]
            pb = df["PB"].iloc[-1]
            pe_pct = df["PE分位"].iloc[-1]
        except:
            if index_code in CACHE_INDEX_VAL:
                return CACHE_INDEX_VAL[index_code]
            pe, pb, pe_pct = 25, 2, 50
        result = {"pe": pe, "pb": pb, "pe_percent": pe_pct}
        CACHE_INDEX_VAL[index_code] = result
        _cache_set(cache_key)
        return result

    @staticmethod
    def get_etf_premium(etf_code: str) -> float:
        time.sleep(REQUEST_DELAY)
        cache_key = f"prem_{etf_code}"
        if etf_code in CACHE_ETF_PREMIUM and _cache_check(cache_key, 3600):
            return CACHE_ETF_PREMIUM[etf_code]
        try:
            df = ak.fund_etf_premium()
            premium = df[df["代码"] == etf_code]["折溢价率"].iloc[0] / 100
        except:
            if etf_code in CACHE_ETF_PREMIUM:
                return CACHE_ETF_PREMIUM[etf_code]
            premium = 0
        CACHE_ETF_PREMIUM[etf_code] = premium
        _cache_set(cache_key)
        return premium

    @staticmethod
    def get_north_flow(index_code: str) -> float:
        time.sleep(REQUEST_DELAY)
        cache_key = f"north_{index_code}"
        if index_code in CACHE_NORTH_CAP and _cache_check(cache_key, 3600):
            return CACHE_NORTH_CAP[index_code]
        try:
            df = ak.stock_hsgt_fund_flow(symbol=index_code)
            flow = df["北向净流入"].tail(5).sum()
        except:
            if index_code in CACHE_NORTH_CAP:
                return CACHE_NORTH_CAP[index_code]
            flow = 0
        CACHE_NORTH_CAP[index_code] = flow
        _cache_set(cache_key)
        return flow
    
    @staticmethod
    def get_margin_balance() -> dict:
        """融资融券余额（两融情绪指标）"""
        cache_key = "margin"
        if hasattr(DataCollectAgent.get_margin_balance, "_cache") and _cache_check(cache_key, 3600):
            return DataCollectAgent.get_margin_balance._cache
        try:
            szse = ak.stock_margin_detail_szse()
            sse = ak.stock_margin_detail_sse()
            result = {
                "szse_margin": float(szse["融资余额"].iloc[-1] / 1e8) if "融资余额" in szse.columns else 0,
                "sse_margin": float(sse["融资余额"].iloc[-1] / 1e8) if "融资余额" in sse.columns else 0,
                "szse_short": float(szse["融券余额"].iloc[-1] / 1e8) if "融券余额" in szse.columns else 0,
            }
        except:
            return {"szse_margin": 0, "sse_margin": 0, "szse_short": 0}
        DataCollectAgent.get_margin_balance._cache = result
        _cache_set(cache_key)
        return result
    
    @staticmethod
    def get_bond_yield() -> dict:
        """中美国债收益率"""
        cache_key = "bond"
        if hasattr(DataCollectAgent.get_bond_yield, "_cache") and _cache_check(cache_key, 7200):
            return DataCollectAgent.get_bond_yield._cache
        try:
            df = ak.bond_zh_us_rate()
            cn10y = float(df[df["指标名称"] == "中国国债收益率10年"]["收益率"].iloc[-1])
            us10y = float(df[df["指标名称"] == "美国国债收益率10年"]["收益率"].iloc[-1])
            result = {"cn_10y": cn10y, "us_10y": us10y, "spread": cn10y - us10y}
        except:
            result = {"cn_10y": 2.5, "us_10y": 4.0, "spread": -1.5}
        DataCollectAgent.get_bond_yield._cache = result
        _cache_set(cache_key)
        return result
    
    @staticmethod
    def get_sector_fund_flow() -> dict:
        """板块资金流向"""
        cache_key = "sector_flow"
        if hasattr(DataCollectAgent.get_sector_fund_flow, "_cache") and _cache_check(cache_key, 3600):
            return DataCollectAgent.get_sector_fund_flow._cache
        try:
            df = ak.stock_sector_fund_flow_summary()
            sector_map = {}
            for _, row in df.iterrows():
                sector_map[row["板块名称"]] = {
                    "流入": float(row.get("主力净流入-净额", 0)),
                    "流入排名": int(row.get("主力净流入-排名", 99))
                }
            result = sector_map
        except:
            result = {}
        DataCollectAgent.get_sector_fund_flow._cache = result
        _cache_set(cache_key)
        return result

    @staticmethod
    def get_ivix(etf_code: str) -> float:
        """获取ETF对应指数的隐含波动率(VIX-like)。3600s TTL，失败返回25.0。"""
        cache_key = f"ivix_{etf_code}"
        if etf_code in CACHE_IVIX and _cache_check(cache_key, 3600):
            return CACHE_IVIX[etf_code]
        try:
            func_map = {
                "510050": ak.index_option_50etf_qvix,
                "510300": ak.index_option_300etf_qvix,
                "510500": ak.index_option_500etf_qvix,
            }
            func = func_map.get(etf_code)
            if func is None:
                val = 25.0
            else:
                df = func()
                val = float(df["收盘价"].iloc[-1]) if "收盘价" in df.columns else float(df.iloc[:, 1].iloc[-1])
        except Exception:
            if etf_code in CACHE_IVIX:
                return CACHE_IVIX[etf_code]
            val = 25.0
        CACHE_IVIX[etf_code] = val
        _cache_set(cache_key)
        return val

    @staticmethod
    def get_futures_basis() -> dict:
        """获取股指期货基差。1800s TTL，失败返回{}。"""
        cache_key = "futures_basis"
        if hasattr(DataCollectAgent.get_futures_basis, "_cache") and _cache_check(cache_key, 1800):
            return DataCollectAgent.get_futures_basis._cache
        try:
            df = ak.futures_zh_realtime()
            contract_map = {"IF": "000300", "IC": "000905", "IH": "000016", "IM": "000852"}
            # 获取指数现货价格
            spot_df = ak.stock_zh_index_spot()
            index_prices = {}
            for col_name in spot_df.columns:
                if "代码" in col_name or "code" in col_name.lower():
                    code_col = col_name
                    break
            else:
                code_col = spot_df.columns[0]
            price_col = "最新价" if "最新价" in spot_df.columns else "current_price"
            for _, row in spot_df.iterrows():
                index_prices[str(row[code_col])] = float(row[price_col]) if price_col in spot_df.columns else 0
            result = {}
            sym_col = "symbol" if "symbol" in df.columns else "代码"
            price_col_f = "current_price" if "current_price" in df.columns else "最新价"
            for prefix in contract_map:
                contracts = df[df[sym_col].astype(str).str.startswith(prefix, na=False)]
                if contracts.empty:
                    continue
                row = contracts.iloc[0]
                fut_price = float(row[price_col_f])
                idx_code = contract_map[prefix]
                idx_price = index_prices.get(idx_code, fut_price)
                basis = (fut_price / idx_price - 1) * 100 if idx_price > 0 else 0
                result[prefix] = round(basis, 2)
        except Exception:
            result = {}
        DataCollectAgent.get_futures_basis._cache = result
        _cache_set(cache_key)
        return result

# ====================== 【2. 宏观研判智能体】 ======================
class MacroAgent:
    @staticmethod
    def analysis() -> tuple[float, str]:
        vol = DataCollectAgent.get_market_total_volume()
        if vol >= 10000:
            return 0.90, f"强趋势行情 | 成交额{vol:.0f}亿 | 全局最大仓位90%"
        elif vol >= 7000:
            return 0.60, f"震荡行情 | 成交额{vol:.0f}亿 | 全局最大仓位60%"
        else:
            return 0.25, f"弱势行情 | 成交额{vol:.0f}亿 | 全局最大仓位25%"

# ====================== 【3. 大模型解读智能体】 ======================
class LLMInterpretAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

    def interpret_news(self, etf_name: str, news_text: str, score: float) -> str:
        """大模型深度解读资讯、情绪与影响"""
        prompt = f"""
        标的：{etf_name}
        相关资讯：{news_text}
        当前舆情情绪分数：{score}（0极度利空，100极度利好）
        请用简短中文解读：1.资讯核心内容 2.市场情绪判断 3.对该标的短期影响，控制在150字内。
        """
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "大模型解读异常，仅参考基础舆情数据"

# ====================== 【4. 核心增强舆情智能体（全优化落地）】 ======================
class PublicOpinionAgent:
    llm_agent = LLMInterpretAgent()

    @staticmethod
    def get_industry_keywords(etf_name: str) -> tuple[set, set]:
        """匹配行业专属关键词"""
        pos_words = set(BASE_POS_KEYWORDS)
        neg_words = set(BASE_NEG_KEYWORDS)
        for industry, words in INDUSTRY_POS.items():
            if industry in etf_name:
                pos_words.update(words)
        for industry, words in INDUSTRY_NEG.items():
            if industry in etf_name:
                neg_words.update(words)
        return pos_words, neg_words

    @staticmethod
    def get_professional_news(keyword: str) -> str:
        """专业财经API优先，失败降级为网页爬虫"""
        # 优先调用专业财经API
        try:
            params = {"key": FIN_API_KEY, "q": keyword, "limit": 3}
            resp = requests.get(FIN_API_URL, params=params, timeout=8)
            data = json.loads(resp.text)
            news = "".join([item["title"] + "。" for item in data["news"]])
            if news:
                return news
        except Exception:
            pass
        # 降级：网页爬虫
        news_content = ""
        url = f"https://finance.sina.com.cn/search/news?q={keyword}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            news_list = soup.find_all("div", class_="result")
            for item in news_list[:3]:
                news_content += item.get_text(strip=True) + "。"
        except Exception:
            news_content = "暂无公开财经资讯"
        return news_content

    @staticmethod
    def extract_keywords(text: str, etf_name: str) -> str:
        """行业专属分词+关键词提取"""
        pos_words, neg_words = PublicOpinionAgent.get_industry_keywords(etf_name)
        words = jieba.lcut(text)
        hit_pos = set(words) & pos_words
        hit_neg = set(words) & neg_words

        if hit_pos and not hit_neg:
            return f"多头关键词：{('、').join(hit_pos)}"
        elif hit_neg and not hit_pos:
            return f"空头关键词：{('、').join(hit_neg)}"
        elif hit_pos and hit_neg:
            return f"多空交织：{('、').join(hit_pos | hit_neg)}"
        else:
            return "无明显情绪关键词"

    @staticmethod
    def sentiment_analysis(text: str) -> float:
        """情感打分 0~100"""
        if not text or "暂无公开财经资讯" in text:
            return 50.0
        s = SnowNLP(text)
        return round(s.sentiments * 100, 2)

    @staticmethod
    def get_opinion_trend(cache_key: str, now_score: float) -> str:
        """舆情趋势跟踪：对比历史分数，判断趋势"""
        today = datetime.now().strftime("%Y%m%d")
        hist_list = CACHE_OPINION_HIST.get(cache_key, [])
        hist_list.append({"date": today, "score": now_score})
        # 仅保留最近N天数据
        hist_list = hist_list[-TREND_DAY_COUNT:]
        CACHE_OPINION_HIST[cache_key] = hist_list

        if len(hist_list) < 2:
            return "趋势不明（数据不足）"
        prev_score = hist_list[-2]["score"]
        diff = now_score - prev_score
        if diff > 8:
            return "舆情持续回暖 ↑"
        elif diff < -8:
            return "舆情持续走弱 ↓"
        else:
            return "舆情平稳震荡 →"

    @staticmethod
    def get_opinion_tag(score: float) -> tuple[str, str, bool]:
        """情绪标签、描述、利空告警"""
        warn_flag = False
        if score >= 70:
            tag = "强利好"
            desc = "市场正面资讯较多，情绪乐观"
        elif score >= 60:
            tag = "偏利好"
            desc = "整体情绪偏积极，关注度向好"
        elif score >= 40:
            tag = "中性舆情"
            desc = "资讯多空均衡，情绪平稳"
        elif score >= OPINION_WARN_THRESHOLD:
            tag = "偏利空"
            desc = "负面资讯增加，市场偏谨慎"
        else:
            tag = "强利空"
            desc = "利空消息集中，风险偏高"
            warn_flag = True
        return tag, desc, warn_flag

    @staticmethod
    def run(etf_name: str, etf_code: str) -> dict:
        time.sleep(REQUEST_DELAY)
        cache_key = f"{etf_code}_{etf_name}"
        if cache_key in CACHE_OPINION:
            return CACHE_OPINION[cache_key]

        # 1. 拉取资讯（API优先 + 爬虫降级）
        news = PublicOpinionAgent.get_professional_news(etf_name)
        # 2. 行业关键词提取
        kw = PublicOpinionAgent.extract_keywords(news, etf_name)
        # 3. 情感打分
        sent_score = PublicOpinionAgent.sentiment_analysis(news)
        # 4. 情绪标签+告警
        op_tag, op_desc, warn_flag = PublicOpinionAgent.get_opinion_tag(sent_score)
        # 5. 舆情趋势分析
        trend = PublicOpinionAgent.get_opinion_trend(cache_key, sent_score)
        # 6. 大模型深度解读
        llm_explain = PublicOpinionAgent.llm_agent.interpret_news(etf_name, news, sent_score)

        res = {
            "news_content": news[:200] + "..." if len(news) > 200 else news,
            "keywords": kw,
            "opinion_score": sent_score,
            "opinion_tag": op_tag,
            "opinion_desc": op_desc,
            "opinion_trend": trend,
            "llm_explain": llm_explain,
            "warn_alert": warn_flag
        }
        CACHE_OPINION[cache_key] = res
        return res

    @staticmethod
    def llm_run(etf_name: str, etf_code: str) -> dict:
        """供LLM Agent调用的舆情接口，不走CACHE（每次新鲜数据）"""
        time.sleep(REQUEST_DELAY)
        news = PublicOpinionAgent.get_professional_news(etf_name)
        kw = PublicOpinionAgent.extract_keywords(news, etf_name)
        sent_score = PublicOpinionAgent.sentiment_analysis(news)
        op_tag, op_desc, warn_flag = PublicOpinionAgent.get_opinion_tag(sent_score)
        cache_key = f"{etf_code}_{etf_name}"
        trend = PublicOpinionAgent.get_opinion_trend(cache_key, sent_score)
        return {
            "news_content": news[:200] + "..." if len(news) > 200 else news,
            "keywords": kw,
            "opinion_score": sent_score,
            "opinion_tag": op_tag,
            "opinion_desc": op_desc,
            "opinion_trend": trend,
            "warn_alert": warn_flag
        }

# ====================== 【LLM多智能体基类】 ======================
class BaseLLMAgent:
    """所有LLM驱动Agent的基类。LLM失败时自动fallback到规则评分。"""
    
    ROLE_NAME = "基础分析师"
    SYSTEM_PROMPT = "你是一个专业的金融分析师。请基于提供的数据进行分析。"
    # 各Agent可覆盖以下配置实现temperature/model多样性
    AGENT_TEMPERATURE = None  # None=使用LLM_TEMPERATURE全局值
    AGENT_MODEL = None        # None=使用LLM_MODEL全局值
    
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        ) if LLM_API_KEY != "your-api-key" else None
    
    def _build_user_prompt(self, etf_name: str, etf_code: str, data_text: str) -> str:
        return f"""标的：{etf_name}（{etf_code}）
时间：{datetime.now().strftime('%Y-%m-%d')}

数据：
{data_text}

请严格按照以下JSON格式输出（不要markdown代码块标记）：
{{
    "rating": "看多/看空/中性/强烈看多/强烈看空",
    "score": 0-100的整数分数,
    "analysis": "你的分析报告（200-400字）",
    "key_factors": ["关键因子1", "关键因子2"],
    "risk_warnings": ["风险点1", "风险点2"],
    "confidence": 0-1之间的置信度
}}"""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict | None:
        if self.client is None or not LLM_ENABLED:
            return None
        try:
            temp = self.AGENT_TEMPERATURE if self.AGENT_TEMPERATURE is not None else LLM_TEMPERATURE
            model = self.AGENT_MODEL if self.AGENT_MODEL is not None else LLM_MODEL
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=LLM_MAX_TOKENS
            )
            text = resp.choices[0].message.content.strip()
            # 清理可能的markdown代码块
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            return json.loads(text)
        except Exception as e:
            print(f"  ⚠️ {self.ROLE_NAME} LLM调用失败: {e}")
            return None
    
    def _parse_to_report(self, etf_code: str, etf_name: str,
                         llm_output: dict | None, fallback_score: float,
                         fallback_rating: str | None = None) -> AgentReport:
        if llm_output is None:
            return AgentReport(
                agent_name=self.ROLE_NAME,
                etf_code=etf_code,
                etf_name=etf_name,
                rating=fallback_rating or self._score_to_rating(fallback_score),
                score=round(fallback_score, 1),
                analysis=f"【规则评分模式】LLM不可用，基于规则模型评分 {fallback_score} 分。",
                key_factors=["规则评分（LLM fallback）"],
                risk_warnings=[],
                confidence=0.5,
                source="rule_fallback"
            )
        
        rating = llm_output.get("rating", "中性")
        if rating not in RATING_ORDER:
            rating = "中性"
        
        return AgentReport(
            agent_name=self.ROLE_NAME,
            etf_code=etf_code,
            etf_name=etf_name,
            rating=rating,
            score=float(np.clip(llm_output.get("score", 50), 0, 100)),
            analysis=llm_output.get("analysis", ""),
            key_factors=llm_output.get("key_factors", []),
            risk_warnings=llm_output.get("risk_warnings", []),
            confidence=float(np.clip(llm_output.get("confidence", 0.5), 0, 1)),
            source="llm"
        )
    
    @staticmethod
    def _score_to_rating(score: float) -> str:
        if score >= 80: return "强烈看多"
        if score >= 65: return "看多"
        if score >= 45: return "中性"
        if score >= 30: return "看空"
        return "强烈看空"


# ====================== 【LLM多智能体 - 宏观分析】 ======================
class MacroAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "宏观分析智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是拥有20年经验的央行宏观分析师。
你的分析框架：
1. 市场成交量代表流动性和参与度——量能决定行情级别
2. 成交额>10000亿=强趋势市场，7000-10000亿=震荡，<7000亿=弱势
3. 给出基于流动性的宏观环境和仓位建议
务必简洁专业，数据驱动。"""
    
    def run(self) -> AgentReport:
        vol = DataCollectAgent.get_market_total_volume()
        data_text = f"今日全市场成交额：{vol:.0f}亿元"
        
        # LLM分析
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("全市场", "MACRO", data_text))
        
        # Fallback: 基于阈值的规则
        if llm_out is None:
            if vol >= 10000:
                score, rating, analysis = 90, "强烈看多", "强趋势行情"
            elif vol >= 7000:
                score, rating, analysis = 60, "看多", "震荡行情"
            else:
                score, rating, analysis = 25, "看空", "弱势行情"
            return AgentReport(
                agent_name=self.ROLE_NAME, etf_code="MACRO", etf_name="全市场",
                rating=rating, score=score,
                analysis=f"【规则评分】{analysis} | 成交额{vol:.0f}亿",
                key_factors=[f"成交额{vol:.0f}亿"],
                risk_warnings=["规则评分（LLM不可用）"],
                confidence=0.6, source="rule_fallback",
                data_summary={"market_volume": vol}
            )
        
        return self._parse_to_report("MACRO", "全市场", llm_out, 
                                      fallback_score=50, fallback_rating="中性")


# ====================== 【LLM多智能体 - 价值估值】 ======================
class ValueAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "价值估值智能体"
    AGENT_TEMPERATURE = 0.3
    SYSTEM_PROMPT = """你是信奉格雷厄姆-巴菲特价值投资理念的分析师。
分析框架：
1. PE/PB百分位是核心——百分位<30%为低估，>70%为高估
2. 低估值+合理百分位=安全边际充足，看好
3. 高估值+高百分位=泡沫风险，看空
4. 百分位在30%-70%之间为估值合理区域
给出基于估值安全边际的判断。"""
    
    def run(self, etf_code: str, etf_name: str, index_code: str) -> AgentReport:
        d = DataCollectAgent.get_index_val(index_code)
        score = round(100 - d["pe_percent"], 2)
        data_text = f"PE(TTM): {d['pe']} | PB: {d['pb']} | PE近5年百分位: {d['pe_percent']}%"
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 技术趋势】 ======================
class TechAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "技术趋势智能体"
    AGENT_TEMPERATURE = 0.3
    SYSTEM_PROMPT = """你是拥有15年经验的技术分析师，擅长趋势识别。
分析框架：
1. 价格与均线关系：价格在MA5和MA20之上=多头排列，之下=空头排列
2. 均线金叉(MA5上穿MA20)=看多信号，死叉=看空信号
3. 近期波动率异常高=风险加大
4. 连续上涨/下跌天数反映短期动能
结合价格位置、均线形态、波动率给出判断。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 20:
            return AgentReport(self.ROLE_NAME, etf_code, etf_name, "中性", 50,
                               "数据不足20个交易日", ["数据不足"], [], 0.3, {}, "rule_fallback")
        close, m5, m20 = df["close"].iloc[-1], df["ma5"].iloc[-1], df["ma20"].iloc[-1]
        ts = 50
        if close > m5 and close > m20: ts += 18
        if m5 > m20: ts += 12
        if close < m5 and close < m20: ts -= 25
        score = float(np.clip(ts, 0, 100))
        recent_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        data_text = (f"最新价: {close:.3f} | MA5: {m5:.3f} | MA20: {m20:.3f}\n"
                     f"均线关系: {'多头' if m5 > m20 else '空头'}排列\n"
                     f"近5日涨跌: {recent_5d:.2f}%\n"
                     f"近5日平均波动率: {df['volatility'].tail(5).mean():.4f}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 舆情情绪】 ======================
class SentimentAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "舆情情绪智能体"
    AGENT_TEMPERATURE = 0.5
    SYSTEM_PROMPT = """你是行为金融学专家，擅长识别市场情绪。
分析框架：
1. 情绪分数>70为乐观（可能过度乐观），<30为恐慌（可能过度悲观）
2. 舆情趋势比单日分数更重要——持续回暖或持续走弱是强烈信号
3. 结合行业关键词判断是否存在实质性利好/利空
4. 关注多空交织情况——矛盾信号意味着市场分歧大
重点关注情绪极端值和趋势变化。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        cache_key = f"{etf_code}_{etf_name}"
        if cache_key in CACHE_OPINION:
            op = CACHE_OPINION[cache_key]
        else:
            op = PublicOpinionAgent.llm_run(etf_name, etf_code)
        data_text = (f"舆情情绪分数: {op['opinion_score']}/100\n"
                     f"情绪标签: {op['opinion_tag']}\n"
                     f"舆情趋势: {op['opinion_trend']}\n"
                     f"情绪关键词: {op['keywords']}\n"
                     f"资讯摘要: {op['news_content'][:300]}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        score = op['opinion_score']
        if score >= 70: fr = "强烈看多"
        elif score >= 60: fr = "看多"
        elif score >= 40: fr = "中性"
        elif score >= 30: fr = "看空"
        else: fr = "强烈看空"
        report = self._parse_to_report(etf_code, etf_name, llm_out, score, fr)
        report.data_summary = {"opinion_score": op['opinion_score'], "opinion_trend": op['opinion_trend'],
                               "keywords": op['keywords'], "warn_alert": op['warn_alert']}
        return report


# ====================== 【LLM多智能体 - 资金流向】 ======================
class FundFlowAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "资金流向智能体"
    AGENT_TEMPERATURE = 0.5
    SYSTEM_PROMPT = """你是专注于资金流分析的市场老手。
分析框架：
1. 北向资金持续流入=外资看好，是重要正向信号
2. 成交量放大(量比>1.2)=资金活跃参与，缩量=观望
3. 折溢价率>1.5%=溢价过高有回落风险
4. 量价配合关系：放量上涨健康，放量下跌危险
结合北向、量比、折溢价综合判断资金面。"""
    
    def run(self, etf_code: str, etf_name: str, index_code: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(10).mean()
        north = DataCollectAgent.get_north_flow(index_code)
        premium = DataCollectAgent.get_etf_premium(etf_code)
        s1 = np.clip(50 + (vol_ratio - 1) * 30, 0, 100)
        s2 = np.clip(50 + north / 100_000_000 * 5, 0, 100)
        score = round((s1 + s2) / 2, 2)
        # 公募股票仓位
        fund_pos_info = ""
        try:
            fp = ak.fund_stock_position_lg()
            if fp is not None and len(fp) > 0:
                fp_val = float(fp.iloc[-1]["股票仓位"])
                fund_pos_info = f"\n公募股票仓位: {fp_val}%"
        except:
            pass
        
        data_text = (f"成交量比(近10日均值): {vol_ratio:.2f}\n"
                     f"北向近5日净流入: {north/100000000:.2f}亿\n"
                     f"折溢价率: {premium*100:.3f}%\n"
                     f"资金分项(s1量比): {s1:.1f} | (s2北向): {s2:.1f}"
                     f"{fund_pos_info}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 风险管理】 ======================
class RiskManagerAgent(BaseLLMAgent):
    ROLE_NAME = "风险管理智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是偏保守的首席风控官，对下行风险极度敏感。
分析框架：
1. 折溢价>1.5%=溢价过高，存在回落风险（扣分）
2. 日波动率>3%=波动剧烈，不适合稳健仓位（扣分）
3. 成交量<5000万=流动性不足，买卖价差大（扣分）
4. 近期最大回撤幅度越大=风险越高
风控视角：宁可错过，不可做错。高评分=低风险。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        premium = DataCollectAgent.get_etf_premium(etf_code)
        vol = df["volatility"].iloc[-1]
        liq = df["volume"].iloc[-1]
        score = 100.0
        risk_detail = []
        if premium > 0.015: score -= 30; risk_detail.append(f"溢价过高({premium*100:.2f}%)")
        if vol > 0.03: score -= 15; risk_detail.append(f"波动率高({vol:.4f})")
        if liq < 5000: score -= 25; risk_detail.append(f"流动性不足(成交量{liq:.0f})")
        score = float(np.clip(score, 0, 100))
        
        # 两融杠杆风险
        margin_note = ""
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_m = margin['szse_margin'] + margin['sse_margin']
            if total_m > 20000: score -= 10; risk_detail.append("两融余额过高，市场杠杆风险大")
            margin_note = f"两融余额: {total_m:.0f}亿"
        except:
            pass
        
        # VIX情绪指标
        vix = DataCollectAgent.get_ivix(etf_code)
        vix_note = f"隐含波动率(VIX): {vix:.1f}"
        if vix > 30:
            score -= 10
            risk_detail.append(f"VIX恐慌({vix:.1f}>30)")
        elif vix < 15:
            score -= 5
            risk_detail.append(f"VIX过低({vix:.1f}<15), 市场可能自满")

        data_text = (f"折溢价率: {premium*100:.3f}%\n"
                     f"日波动率: {vol:.4f}\n"
                     f"成交量: {liq:.0f}\n"
                     f"{vix_note}\n"
                     f"{margin_note}\n"
                     f"风险扣分原因: {'; '.join(risk_detail) if risk_detail else '无明显风险'}")
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 行业纵析】 ======================
class IndustryAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "行业纵析智能体"
    AGENT_TEMPERATURE = 0.4
    SYSTEM_PROMPT = """你是深耕行业的资深研究员，精通行业轮动分析。
分析框架：
1. 识别ETF所属行业赛道（从名称判断）
2. 分析行业所处周期位置
3. 匹配行业专属关键词判断当前景气度
4. 相对强度——该ETF近期相对大盘的表现强弱
5. 行业动量——该ETF近期的涨幅趋势

A股行业轮动特征：
- 强势行业持续期通常3-6个月
- 资金会从高位行业流向低位行业（均值回归）
- 政策催化可以提前启动行业轮动
给出基于行业基本面和轮动位置的独立判断。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        pos_words, neg_words = PublicOpinionAgent.get_industry_keywords(etf_name)
        matched_industries = [ind for ind in INDUSTRY_POS if ind in etf_name]
        industry = matched_industries[0] if matched_industries else "通用"
        
        df = DataCollectAgent.get_etf_price(etf_code)
        close = df["close"].iloc[-1]
        
        ret_20d = (close / df["close"].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        ret_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        
        # 相对强度：对比沪深300
        relative_strength = 0.0
        strength_desc = "无法计算相对强度"
        try:
            benchmark = DataCollectAgent.get_etf_price("510300")
            bench_close = benchmark["close"].iloc[-1]
            bench_ret_20d = (bench_close / benchmark["close"].iloc[-20] - 1) * 100 if len(benchmark) >= 20 else 0
            relative_strength = ret_20d - bench_ret_20d
            strength_desc = f"跑赢大盘{relative_strength:.1f}%" if relative_strength > 0 else f"跑输大盘{abs(relative_strength):.1f}%"
        except Exception:
            pass
        
        # 板块资金流向
        sector_flow_info = ""
        try:
            sector_flow = DataCollectAgent.get_sector_fund_flow()
            if industry in sector_flow:
                sf = sector_flow[industry]
                sector_flow_info = f"板块资金净流入: {sf['流入']/1e8:.1f}亿 | 排名: {sf['流入排名']}"
            else:
                sector_flow_info = "暂无该行业板块资金流向数据"
        except:
            pass
        
        data_text = (f"所属行业: {industry}\n"
                     f"行业正向词: {', '.join(pos_words)}\n"
                     f"行业负向词: {', '.join(neg_words)}\n"
                     f"近5日涨幅: {ret_5d:.2f}%\n"
                     f"近20日涨幅: {ret_20d:.2f}%\n"
                     f"相对强度(20日vs沪深300): {strength_desc}\n"
                     f"{sector_flow_info}")
        
        # 规则评分：行业归属+动量贡献
        score = 50.0
        if industry != "通用": score += 10
        score += float(np.clip(ret_20d * 0.5, -20, 20))
        score = float(np.clip(score, 0, 100))
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 货币政策/流动性】 ======================
class MonetaryPolicyAgent(BaseLLMAgent):
    ROLE_NAME = "货币政策智能体"
    AGENT_TEMPERATURE = 0.15
    SYSTEM_PROMPT = """你是前央行研究员，专注于货币政策与流动性分析。
分析框架：
1. 政策利率（MLF、LPR、7天逆回购利率）变动趋势
2. 存款准备金率(RRR)水平
3. 社融/M2增速
4. 银行间质押式回购利率(DR007)
5. 人民币汇率(USDCNY)对宽松空间的制约
请基于可获得的数据给出流动性环境判断。"""
    
    def run(self) -> AgentReport:
        try:
            df_lpr = ak.interest_rate_lpr()
            lpr_1y = float(df_lpr[df_lpr["期限种类"] == "1年期"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
            lpr_5y = float(df_lpr[df_lpr["期限种类"] == "5年期以上"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
        except Exception:
            lpr_1y, lpr_5y = None, None
        
        try:
            fx = ak.spot_quote()
            usdcny = None
            if fx is not None:
                usd_row = fx[fx["名称"].str.contains("美元", na=False)]
                if len(usd_row) > 0:
                    usdcny = float(usd_row["现价"].iloc[0])
        except Exception:
            usdcny = None
        
        score = 50.0
        signals = []
        
        if lpr_1y is not None:
            signals.append(f"1年期LPR: {lpr_1y}%")
            if lpr_1y <= 3.1: score += 20
            elif lpr_1y >= 3.85: score -= 15
        
        if lpr_5y is not None:
            signals.append(f"5年期LPR: {lpr_5y}%")
        
        if usdcny is not None:
            signals.append(f"USDCNY: {usdcny}")
            if usdcny > 7.3: score -= 15
            elif usdcny < 6.8: score += 10
        
        # 加入中美利差
        try:
            bond = DataCollectAgent.get_bond_yield()
            signals.append(f"中美利差: {bond['spread']}% (中国{bond['cn_10y']}%-美国{bond['us_10y']}%)")
        except:
            pass
        
        # 加入融资融券（两融情绪）
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_margin = margin['szse_margin'] + margin['sse_margin']
            signals.append(f"两融余额: {total_margin:.0f}亿 | 融券: {margin['szse_short']:.0f}亿")
            if total_margin > 15000: score += 5
        except:
            pass
        
        # 宏观数据：M2/SHIBOR
        try:
            m2 = ak.macro_china_m2_yearly()
            if m2 is not None and len(m2) > 0:
                m2_val = float(m2.iloc[-1]["同比增速"])
                signals.append(f"M2同比: {m2_val}%")
                if m2_val > 10: score += 5
        except:
            pass
        try:
            shibor = ak.macro_china_shibor_all()
            if shibor is not None and len(shibor) > 0:
                on_rate = float(shibor.iloc[-1]["ON"]) if "ON" in shibor.columns else 0
                signals.append(f"SHIBOR隔夜: {on_rate}%")
        except:
            pass
        try:
            pmi_df = ak.macro_china_pmi()
            if pmi_df is not None:
                pmi_val = float(pmi_df.tail(1).values[0][1])
                signals.append(f"制造业PMI: {pmi_val}")
                if pmi_val > 52: score += 10
                elif pmi_val < 48: score -= 10
        except:
            pass
        
        score = float(np.clip(score, 0, 100))
        data_text = "\n".join(signals) if signals else "暂无实时货币政策数据"
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国市场", "MONETARY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("MONETARY", "货币政策", llm_out, score, rating)
        report.data_summary = {"lpr_1y": lpr_1y, "lpr_5y": lpr_5y, "usdcny": usdcny}
        return report


# ====================== 【LLM多智能体 - 政策事件/政治周期】 ======================
class PolicyEventAgent(BaseLLMAgent):
    ROLE_NAME = "政策事件智能体"
    AGENT_TEMPERATURE = 0.6
    SYSTEM_PROMPT = """你是资深政策分析师，专注中国政治经济周期。
分析框架：
1. 识别当前所处的政策周期阶段
2. 重大会议窗口：两会(3月)、政治局会议(4/7/10/12月)、中央经济工作会议(12月)
3. 行业政策催化——近期是否有针对特定行业的重大政策出台
4. 政策基调判断：宽松/中性/收紧
A股特征：两会前后春季躁动，政治局会议定调影响季度级别方向。"""
    
    # 重要会议日历 (月, 开始日, 结束日)
    KEY_EVENTS = [
        ("两会", 3, 1, 3, 15),
        ("政治局会议(4月)", 4, 15, 4, 30),
        ("政治局会议(7月)", 7, 15, 7, 31),
        ("政治局会议(10月)", 10, 15, 10, 31),
        ("中央经济工作会议", 12, 1, 12, 15),
    ]
    
    def run(self) -> AgentReport:
        now = datetime.now()
        month, day = now.month, now.day
        
        current_event = None
        for name, sm, sd, em, ed in self.KEY_EVENTS:
            try:
                ev_start = datetime(now.year, sm, sd)
                ev_end = datetime(now.year, em, ed)
                if ev_start <= now <= ev_end:
                    current_event = f"当前处于{name}窗口期"
                    break
                if now < ev_start:
                    days_to = (ev_start - now).days
                    if days_to <= 30:
                        current_event = f"距离{name}还有{days_to}天"
                        break
            except Exception:
                continue
        
        season_effect = ""
        if month in (1, 2, 3): season_effect = "春季躁动窗口"
        elif month == 4: season_effect = "年报季+政治局会议定调"
        elif month in (7, 8): season_effect = "中报季+政治局会议落地"
        elif month in (10, 11): season_effect = "三季报+年末政策定调"
        elif month == 12: season_effect = "中央经济工作会议+机构调仓"
        
        data_text = (f"当前日期: {now.strftime('%Y-%m-%d')}\n"
                     f"季节效应: {season_effect}\n"
                     f"会议窗口: {current_event or '无重要会议窗口'}\n"
                     f"月份特征: {month}月")
        
        score = 55.0
        if current_event: score += 10
        if month in (3, 7, 12): score += 8
        if month in (1, 2): score += 5
        score = float(np.clip(score, 0, 100))
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国政策周期", "POLICY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("POLICY", "政策周期", llm_out, score, rating)
        report.data_summary = {"current_event": current_event, "season_effect": season_effect, "month": month}
        return report


# ====================== 【LLM多智能体 - 零售情绪结构】 ======================
class RetailSentimentAgent(BaseLLMAgent):
    ROLE_NAME = "零售情绪智能体"
    AGENT_TEMPERATURE = 0.5
    SYSTEM_PROMPT = """你是行为金融学量化分析师，专注A股散户情绪测量。
分析框架：
1. 量比（成交量/20日均量）——放量过大=情绪过热，缩量=冷清
2. 价格位置（52周高低位置）——接近高点=亢奋，接近低点=恐慌
3. 散户情绪是典型的反向指标——极度乐观时见顶，极度悲观时见底
请基于量价数据给出情绪判断。"""
    
    def run(self, etf_code: str) -> AgentReport:
        df = DataCollectAgent.get_etf_price(etf_code)
        
        vol_20_mean = df["volume"].tail(20).mean()
        vol_today = df["volume"].iloc[-1]
        turnover_ratio = vol_today / vol_20_mean if vol_20_mean > 0 else 1.0
        
        close = df["close"].iloc[-1]
        high_52w = df["close"].tail(250).max() if len(df) >= 250 else df["close"].max()
        low_52w = df["close"].tail(250).min() if len(df) >= 250 else df["close"].min()
        pos_from_low = (close - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        
        score = 50.0
        signals = []
        
        if turnover_ratio > 2.0:
            signals.append(f"放量异常(量比{turnover_ratio:.2f})"); score -= 10
        elif turnover_ratio > 1.5:
            signals.append(f"放量(量比{turnover_ratio:.2f})"); score += 5
        elif turnover_ratio < 0.5:
            signals.append(f"缩量(量比{turnover_ratio:.2f})"); score -= 5
        else:
            signals.append(f"量能正常(量比{turnover_ratio:.2f})")
        
        if pos_from_low > 90:
            signals.append("接近年内高点"); score -= 8
        elif pos_from_low < 10:
            signals.append("接近年内低点"); score += 8
        elif 40 <= pos_from_low <= 60:
            signals.append("价格中位区间"); score += 3
        
        # 加入两融数据
        margin_info = ""
        try:
            margin = DataCollectAgent.get_margin_balance()
            total_m = margin['szse_margin'] + margin['sse_margin']
            margin_info = f"两融余额: {total_m:.0f}亿"
        except:
            pass
        
        # 市场活跃度数据
        activity_info = ""
        try:
            act = ak.stock_market_activity_legu()
            if act is not None and "换手率" in act.columns:
                turnover_rate = float(act["换手率"].iloc[-1])
                activity_info = f"\n全市场换手率: {turnover_rate}%"
        except:
            pass
        
        score = float(np.clip(score, 0, 100))
        data_text = (f"量比(20日均值): {turnover_ratio:.2f}\n"
                     f"52周价格位置: {pos_from_low:.1f}%\n"
                     f"情绪信号: {'; '.join(signals)}\n"
                     f"{margin_info}"
                     f"{activity_info}")
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("市场情绪", etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report(etf_code, "市场情绪", llm_out, score, rating)
        report.data_summary = {"turnover_ratio": turnover_ratio, "pos_from_low": pos_from_low}
        return report


# ====================== 【LLM多智能体 - 跨资产/跨市场联动】 ======================
class CrossMarketAgent(BaseLLMAgent):
    ROLE_NAME = "跨市场联动智能体"
    AGENT_TEMPERATURE = 0.6
    SYSTEM_PROMPT = """你是全球宏观策略分析师，专注跨市场信号传导。
分析框架：
1. 人民币汇率(USDCNY)——升值利好A股，贬值承压
2. 美债收益率——全球资产定价锚
3. 美元指数——美元强弱影响外资流向
4. 黄金价格——避险情绪指标
传导逻辑：美债利率+美元同时走强=新兴市场承压；人民币升值+美元弱=利好A股。"""
    
    def run(self, etf_name: str, etf_code: str) -> AgentReport:
        signals = []
        usdcny = None
        try:
            fx = ak.spot_quote()
            cny_row = fx[fx["名称"].str.contains("美元", na=False)]
            if len(cny_row) > 0:
                usdcny = float(cny_row["现价"].iloc[0])
                signals.append(f"USDCNY: {usdcny}")
        except Exception:
            pass
        
        # 加入中美利差数据
        try:
            bond = DataCollectAgent.get_bond_yield()
            signals.append(f"中国10Y国债: {bond['cn_10y']}% | 美国10Y国债: {bond['us_10y']}% | 中美利差: {bond['spread']}%")
        except:
            pass
        
        score = 50.0
        if usdcny is not None:
            if usdcny > 7.3: score -= 15
            elif usdcny < 6.9: score += 10

        # 巴菲特指数（全市场市值/GDP，估值温度计）
        try:
            buffett = ak.stock_buffett_index_lg()
            if buffett is not None and len(buffett) > 0:
                b_val = float(buffett["value"].iloc[-1])
                signals.append(f"巴菲特指数: {b_val:.0f}%")
                if b_val > 100: score -= 10
                elif b_val < 60: score += 10
        except:
            pass

        # 股指期货基差
        try:
            basis_data = DataCollectAgent.get_futures_basis()
            if basis_data:
                basis_parts = [f"{k}基差: {v:+.2f}%" for k, v in basis_data.items()]
                signals.append(" | ".join(basis_parts))
                avg_basis = sum(basis_data.values()) / len(basis_data)
                if avg_basis > 0:
                    score += 5
                    signals.append(f"平均基差{avg_basis:+.2f}%, 升水(contango)偏多")
                else:
                    score -= 5
                    signals.append(f"平均基差{avg_basis:+.2f}%, 贴水(backwardation)偏空")
        except:
            pass

        score = float(np.clip(score, 0, 100))
        data_text = "\n".join(signals) if signals else "暂无实时跨市场数据"
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【辩论引擎】 ======================
class DebateEngine:
    """矛盾检测 + 选择性辩论（LLM驱动）"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        ) if LLM_API_KEY != "your-api-key" else None
    
    @staticmethod
    def _call_llm(system_prompt: str, user_prompt: str) -> str | None:
        if LLM_API_KEY == "your-api-key" or not LLM_ENABLED:
            return None
        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ DebateEngine LLM调用失败: {e}")
            return None
    
    @staticmethod
    def detect_disagreements(reports: list[AgentReport]) -> list[dict]:
        """检测Agent间是否存在显著分歧"""
        disagreements = []
        agent_names = [r.agent_name for r in reports]
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a1, a2 = reports[i], reports[j]
                score_gap = abs(a1.score - a2.score)
                rating_gap = abs(RATING_ORDER.index(a1.rating) - RATING_ORDER.index(a2.rating))
                if score_gap >= DISAGREEMENT_SCORE_THRESHOLD or rating_gap >= DISAGREEMENT_RATING_GAP:
                    disagreements.append({
                        "agent_a": a1.agent_name, "agent_b": a2.agent_name,
                        "rating_a": a1.rating, "rating_b": a2.rating,
                        "score_a": a1.score, "score_b": a2.score,
                        "score_gap": score_gap,
                        "focus": f"{a1.agent_name}({a1.rating}) vs {a2.agent_name}({a2.rating}) 存在分歧"
                    })
        return disagreements
    
    @staticmethod
    def hold_debate(disagreements: list[dict], reports: list[AgentReport],
                    etf_name: str, etf_code: str) -> list[dict]:
        """执行选择性辩论：对分歧双方进行真实LLM驱动辩论"""
        if not DEBATE_ENABLED or not disagreements:
            return []
        debate_logs = []
        reports_dict = {r.agent_name: r for r in reports}
        for d in disagreements[:3]:
            agent_a = reports_dict.get(d["agent_a"])
            agent_b = reports_dict.get(d["agent_b"])
            if not agent_a or not agent_b:
                continue
            debate_log = {"topic": d["focus"], "agents": [d["agent_a"], d["agent_b"]], "rounds": []}

            if LLM_ENABLED and LLM_API_KEY != "your-api-key":
                sys_prompt = f"""你是ETF投研辩论主持人。以下两位分析师对{etf_name}({etf_code})存在分歧。
请基于他们的实际分析内容，生成真实的辩论对话。保持专业、数据驱动的风格，每个发言控制在150字以内。"""

                r1_prompt = f"""分析师A（{d['agent_a']}）评级{d['rating_a']}({d['score_a']}分)，完整分析：{agent_a.analysis}

分析师B（{d['agent_b']}）评级{d['rating_b']}({d['score_b']}分)，完整分析：{agent_b.analysis}

请生成第1轮辩论：
1. {d['agent_a']}陈述自己的核心观点和论据
2. {d['agent_b']}针对{d['agent_a']}的观点进行反驳，引用自己的分析数据

以JSON格式输出（不要markdown代码块标记）：
{{"statement": "A的陈述", "rebuttal": "B的反驳"}}"""

                r1_text = DebateEngine._call_llm(sys_prompt, r1_prompt)
                r1_data = None
                if r1_text:
                    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', r1_text.strip())
                    try:
                        r1_data = json.loads(cleaned)
                    except Exception:
                        pass

                if r1_data and "statement" in r1_data and "rebuttal" in r1_data:
                    debate_log["rounds"].append({
                        "round": 1,
                        "statement": f"{d['agent_a']}: {r1_data['statement']}",
                        "rebuttal": f"{d['agent_b']}: {r1_data['rebuttal']}"
                    })
                else:
                    debate_log["rounds"].append({
                        "round": 1,
                        "statement": f"{d['agent_a']}: 评级{d['rating_a']}({d['score_a']}分)。{agent_a.analysis[:200]}",
                        "rebuttal": f"{d['agent_b']}: 评级{d['rating_b']}({d['score_b']}分)。{agent_b.analysis[:200]}"
                    })

                if DEBATE_ROUNDS >= 2:
                    r2_prompt = f"""基于第1轮辩论：
A: {debate_log['rounds'][0]['statement']}
B: {debate_log['rounds'][0]['rebuttal']}

分析师A（{d['agent_a']}）关键因子：{', '.join(agent_a.key_factors)}
分析师B（{d['agent_b']}）关键因子：{', '.join(agent_b.key_factors)}

请生成第2轮辩论：
1. {d['agent_b']}针对{d['agent_a']}的论据提出质疑和反问
2. {d['agent_a']}回应{d['agent_b']}的质疑，维护自己的观点

以JSON格式输出（不要markdown代码块标记）：
{{"challenge": "B的质疑", "response": "A的回应"}}"""

                    r2_text = DebateEngine._call_llm(sys_prompt, r2_prompt)
                    r2_data = None
                    if r2_text:
                        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', r2_text.strip())
                        try:
                            r2_data = json.loads(cleaned)
                        except Exception:
                            pass

                    if r2_data and "challenge" in r2_data and "response" in r2_data:
                        debate_log["rounds"].append({
                            "round": 2,
                            "challenge": f"{d['agent_b']}: {r2_data['challenge']}",
                            "response": f"{d['agent_a']}: {r2_data['response']}"
                        })
                    else:
                        key_b = ', '.join(agent_b.key_factors[:2]) if agent_b.key_factors else '潜在风险'
                        key_a = ', '.join(agent_a.key_factors[:3]) if agent_a.key_factors else '多维度数据'
                        debate_log["rounds"].append({
                            "round": 2,
                            "challenge": f"{d['agent_b']}反问：{d['agent_a']}的{d['rating_a']}判断是否低估了{key_b}等因素的影响？",
                            "response": f"{d['agent_a']}回应：{key_a}支撑我的判断，{d['agent_b']}提示的风险已标注在报告的风险提示中。"
                        })
            else:
                key_b = ', '.join(agent_b.key_factors[:2]) if agent_b.key_factors else '潜在风险'
                key_a = ', '.join(agent_a.key_factors[:3]) if agent_a.key_factors else '多维度数据'
                debate_log["rounds"].append({
                    "round": 1,
                    "statement": f"{d['agent_a']}: 评级{d['rating_a']}({d['score_a']}分)。核心逻辑：{agent_a.analysis[:200]}",
                    "rebuttal": f"{d['agent_b']}: 评级{d['rating_b']}({d['score_b']}分)。核心逻辑：{agent_b.analysis[:200]}"
                })
                if DEBATE_ROUNDS >= 2:
                    debate_log["rounds"].append({
                        "round": 2,
                        "challenge": f"{d['agent_b']}反问：{d['agent_a']}的{d['rating_a']}判断是否忽略了{key_b}？",
                        "response": f"{d['agent_a']}回应：{key_a}支持我的判断，{d['agent_b']}提醒的风险点已记录。"
                    })
            debate_logs.append(debate_log)
        return debate_logs


# ====================== 【5. 分项打分智能体】 ======================
class ValueScoreAgent:
    @staticmethod
    def run(index_code: str) -> float:
        d = DataCollectAgent.get_index_val(index_code)
        return round(100 - d["pe_percent"], 2)

class BoomScoreAgent:
    @staticmethod
    def run(etf_code: str) -> float:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 20:
            return 50.0
        chg = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
        return float(np.clip(50 + chg * 2, 0, 100))

class TechScoreAgent:
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta).clip(lower=0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def _macd_histogram(series: pd.Series) -> float:
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return float(hist.iloc[-1])

    @staticmethod
    def _bollinger_pct_b(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> float:
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        pct_b = (series - lower) / (upper - lower)
        return float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else 0.5

    @staticmethod
    def run(etf_code: str) -> float:
        df = DataCollectAgent.get_etf_price(etf_code)
        if len(df) < 20:
            return 50.0
        c, m5, m20 = df["close"].iloc[-1], df["ma5"].iloc[-1], df["ma20"].iloc[-1]
        score = 50
        if c > m5 and c > m20:
            score += 18
        if m5 > m20:
            score += 12
        if c < m5 and c < m20:
            score -= 25

        rsi = TechScoreAgent._rsi(df["close"], 14)
        if rsi < 30:
            score += 10
        elif rsi > 70:
            score -= 8

        macd_hist = TechScoreAgent._macd_histogram(df["close"])
        if macd_hist > 0:
            score += 8
        else:
            score -= 8

        pct_b = TechScoreAgent._bollinger_pct_b(df["close"])
        if pct_b < 0:
            score += 5
        elif pct_b > 1:
            score -= 5

        return float(np.clip(score, 0, 100))

class FundScoreAgent:
    @staticmethod
    def run(etf_code: str, index_code: str) -> float:
        df = DataCollectAgent.get_etf_price(etf_code)
        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(10).mean()
        north = DataCollectAgent.get_north_flow(index_code)
        s1 = np.clip(50 + (vol_ratio - 1) * 30, 0, 100)
        s2 = np.clip(50 + north / 100000000 * 5, 0, 100)
        return round((s1 + s2) / 2, 2)

class RiskScoreAgent:
    @staticmethod
    def run(etf_code: str) -> float:
        df = DataCollectAgent.get_etf_price(etf_code)
        premium = DataCollectAgent.get_etf_premium(etf_code)
        vol = df["volatility"].iloc[-1]
        liq = df["volume"].iloc[-1]
        score = 100
        if premium > PREMIUM_RISK_THRESHOLD:
            score -= 30
        if vol > VOL_RISK_THRESHOLD:
            score -= 15
        if liq < LIQ_THRESHOLD:
            score -= 25
        return float(np.clip(score, 0, 100))

class BacktestAgent:
    @staticmethod
    def run(etf_code: str, days=120) -> dict:
        snapshot_dir = "data/snapshots"
        files = sorted(glob.glob(f"{snapshot_dir}/*.json"))
        if len(files) < 2:
            return BacktestAgent._fallback(etf_code, days)

        df = DataCollectAgent.get_etf_price(etf_code)
        ds = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        df_dates = set(ds)

        dates, signals = [], []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                snap = json.load(f)
            for etf in snap["etfs"]:
                if etf["code"] == etf_code:
                    dates.append(snap["date"])
                    signals.append(etf)
                    break

        if len(dates) < 2:
            return BacktestAgent._fallback(etf_code, days)

        date_idx = dict(zip(ds, range(len(df))))

        for sig_date in dates:
            if sig_date not in df_dates:
                return BacktestAgent._fallback(etf_code, days)

        strategy_rets = []
        for i, sig_date in enumerate(dates):
            idx = date_idx[sig_date]
            if idx + 1 >= len(df):
                continue
            pos = 1.0 if signals[i].get("position_pct", 0) > 0 else 0.0
            ret = df.iloc[idx + 1]["close"] / df.iloc[idx]["close"] - 1
            strategy_rets.append(pos * ret)

        if len(strategy_rets) < 2:
            return BacktestAgent._fallback(etf_code, days)

        arr = np.array(strategy_rets)
        win = np.sum(arr > 0)
        lose = np.sum(arr < 0)
        win_rate = win / (win + lose) if (win + lose) > 0 else 0
        profit_avg = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0
        loss_avg = abs(float(np.mean(arr[arr < 0]))) if np.any(arr < 0) else 0
        pl_ratio = profit_avg / loss_avg if loss_avg > 0 else 1
        cum = np.cumprod(1 + arr)
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
        sharpe = float(np.mean(arr) / max(np.std(arr), 1e-10) * np.sqrt(252))

        return {
            "胜率": round(win_rate * 100, 2),
            "盈亏比": round(pl_ratio, 2),
            "最大回撤": round(max_dd * 100, 2),
            "回测收益": round((cum[-1] - 1) * 100, 2)
        }

    @staticmethod
    def _fallback(etf_code: str, days=120) -> dict:
        df = DataCollectAgent.get_etf_price(etf_code).tail(days).copy()
        df["ma20"] = df["close"].rolling(20).mean()
        df["signal"] = (df["close"] > df["ma20"]).astype(int)
        df["ret"] = df["close"].pct_change()
        df["strategy_ret"] = df["signal"].shift(1) * df["ret"]

        win = len(df[df["strategy_ret"] > 0])
        lose = len(df[df["strategy_ret"] < 0])
        win_rate = win / (win + lose) if (win + lose) > 0 else 0
        profit_avg = df[df["strategy_ret"] > 0]["strategy_ret"].mean() or 0
        loss_avg = abs(df[df["strategy_ret"] < 0]["strategy_ret"].mean() or 0)
        pl_ratio = profit_avg / loss_avg if loss_avg > 0 else 1
        cum = (1 + df["strategy_ret"]).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        return {
            "胜率": round(win_rate * 100, 2),
            "盈亏比": round(pl_ratio, 2),
            "最大回撤": round(max_dd * 100, 2),
            "回测收益": round((cum.iloc[-1] - 1) * 100, 2)
        }

# ====================== 【首席决策智能体】 ======================
class ChiefDecisionAgent(BaseLLMAgent):
    ROLE_NAME = "首席决策智能体"
    AGENT_TEMPERATURE = 0.15
    
    # 评分Agent列表（用于动态权重和方向校准）
    SCORING_AGENTS = [
        "价值估值智能体", "技术趋势智能体", "舆情情绪智能体", "资金流向智能体",
        "风险管理智能体", "行业纵析智能体", "零售情绪智能体", "跨市场联动智能体"
    ]
    # 风险管理Agent是"反向"的——高分=安全，需要反转
    REVERSE_AGENTS = {"风险管理智能体"}
    
    SYSTEM_PROMPT = """你是投资委员会主席，需要综合各方观点做出最终判断。
你的职责：
1. 阅读所有智能体的独立分析报告
2. 审阅辩论记录（如有）
3. 综合不同维度的观点，考虑每个Agent的置信度和专业性
4. 对分歧点做出仲裁判断
5. 给出明确的最终评级、仓位建议和核心逻辑

参考决策示例：
示例1：Agent报告=[价值:看空(35分), 技术:看多(70分), 情绪:看多(65分), 资金:中性(50分), 风控:安全(90分)] 辩论=[技术vs价值关于估值分歧]
→ 最终: 看多(62分) 逻辑=技术面强势但估值偏高，折中判断有限看多，仓位中等
示例2：Agent报告=[全部看多, 75-90分] 辩论=无分歧
→ 最终: 强烈看多(85分) 逻辑=多维度共振，高置信度看多，仓位重仓
示例3：Agent报告=[价值:看空(30分), 技术:看空(25分), 情绪:中性(50分), 资金:看空(35分)] 辩论=无
→ 最终: 看空(32分) 逻辑=多维度一致偏弱，回避风险

评级标准：
- 强烈看多 (score>=80)：多项指标共振，核心机会
- 看多 (score>=65)：整体向好，有少量顾虑
- 中性 (score>=45)：多空均衡，等待信号
- 看空 (score>=30)：整体偏弱，谨慎回避
- 强烈看空 (score<30)：多项风险暴露，清仓回避"""
    
    @staticmethod
    def load_agent_weights() -> dict:
        """从ReviewManager历史准确率加载动态权重"""
        weights = {name: 1.0 for name in ChiefDecisionAgent.SCORING_AGENTS}
        try:
            if os.path.exists(ReviewManager.REVIEW_FILE):
                with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                for aname, data in stats.get("by_agent", {}).items():
                    d = data["directional"]
                    h = data["hits"]
                    if d > 2:  # 至少2次即可（降冷启动门槛）
                        acc = h / d
                        weights[aname] = 0.3 + acc * 1.4  # 0.3~1.7范围，0.5准确率=1.0中性
        except:
            pass
        return weights
    
    @staticmethod
    def _compute_zscore(scores: list[float]) -> list[float]:
        """z-score标准化"""
        arr = np.array(scores)
        mean, std = np.mean(arr), np.std(arr)
        if std < 1e-6:
            return [0.0] * len(scores)
        return [float((s - mean) / std) for s in scores]
    
    @staticmethod
    def _sigmoid_penalty(value: float, threshold: float, slope: float = 500) -> float:
        """平滑过渡惩罚——使用sigmoid替代硬切断"""
        import math
        return 1.0 / (1.0 + math.exp(-slope * (value - threshold)))
    
    @staticmethod
    def _kelly_position(win_rate: float, avg_win: float, avg_loss: float, max_pos: float) -> float:
        """简化凯利公式仓位计算"""
        if avg_loss <= 0 or win_rate >= 1:
            return max_pos
        b = avg_win / abs(avg_loss)  # 赔率
        p = win_rate
        q = 1 - p
        if b <= 0:
            return 0
        f_star = (p * b - q) / b
        return float(np.clip(f_star * max_pos, 0.05, max_pos))
    
    @staticmethod
    def _time_series_momentum_score(etf_code: str) -> float:
        try:
            df = DataCollectAgent.get_etf_price(etf_code)
            if len(df) < 20:
                return 0.0
            close = df["close"].values[-20:]
            x = np.arange(20)
            A = np.vstack([x, np.ones(20)]).T
            slope, intercept = np.linalg.lstsq(A, close, rcond=None)[0]
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            trend_strength = r2 * (1 if slope > 0 else -1) * 4.0

            consistency = 0
            ma20_series = df["close"].rolling(20).mean()
            for i in range(-20, 0):
                if df["close"].iloc[i] > ma20_series.iloc[i]:
                    consistency += 1
                else:
                    consistency -= 1
            consistency_score = consistency / 20.0 * 4.0

            adx_signal = 0.0
            if "high" in df.columns and "low" in df.columns and len(df) >= 35:
                high = df["high"].values[-35:]
                low = df["low"].values[-35:]
                close_a = df["close"].values[-35:]
                tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close_a[:-1]))
                tr = np.maximum(tr, np.abs(low[1:] - close_a[:-1]))
                atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                up_move = high[1:] - high[:-1]
                down_move = low[:-1] - low[1:]
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                di_plus = np.sum(plus_dm[-14:]) / max(atr_14, 1e-10) * 100
                di_minus = np.sum(minus_dm[-14:]) / max(atr_14, 1e-10) * 100
                dx = abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-10) * 100
                if di_plus > di_minus:
                    adx_signal = dx / 100.0 * 2.0
                else:
                    adx_signal = -dx / 100.0 * 2.0

            score = trend_strength + consistency_score + adx_signal
            return float(np.clip(score, -10, 10))
        except Exception:
            return 0.0

    def run(self, reports: list[AgentReport], debates: list[dict],
            global_max_pos: float, etf_info: dict,
            market_state: str = "震荡偏强") -> FinalResearchReport:
        
        # ── 1. 评分预处理：RiskAgent反转 + z-score标准化 ──
        adjusted_scores = []
        for r in reports:
            score = r.score
            if r.agent_name in self.REVERSE_AGENTS:
                score = 100 - score  # 风控反转：100=安全→0=安全，0=高风险→100=高风险
            adjusted_scores.append(score)
        
        # z-score标准化
        z_scores = self._compute_zscore(adjusted_scores)
        # 转回0-100分制（z-score → 50 + z*15）
        normalized_scores = [float(np.clip(50 + z * 15, 0, 100)) for z in z_scores]
        
        # ── 2. 动态加权平均 ──
        weights = self.load_agent_weights()
        weight_values = []
        for r in reports:
            w = weights.get(r.agent_name, 1.0)
            # 附加置信度加权
            w *= (0.5 + r.confidence)
            weight_values.append(w)
        
        # 因子贡献追踪：(normalized_score - 50) * weight_factor
        factor_contributions = {}
        for r, ns, wv in zip(reports, normalized_scores, weight_values):
            factor_contributions[r.agent_name] = round((ns - 50) * wv, 2)
        
        total_w = sum(weight_values)
        if total_w > 0:
            norm_weights = [w / total_w for w in weight_values]
            weighted_score = sum(n * s for n, s in zip(norm_weights, normalized_scores))
        else:
            weighted_score = np.mean(normalized_scores)

        # ── 2.5 时间序列动量因子调整 ──
        ts_momentum = self._time_series_momentum_score(etf_info['code'])
        weighted_score += ts_momentum

        # ── 3. 共识度计算（连续版） ──
        score_std = float(np.std(normalized_scores))
        score_mean = float(np.mean(normalized_scores))
        cv = score_std / max(score_mean, 1)  # 变异系数
        
        if cv < 0.1:
            consensus = "高度一致"
        elif cv < 0.2:
            consensus = "基本一致"
        elif cv < 0.35:
            consensus = "存在分歧"
        else:
            consensus = "严重分歧"
        
        # 分歧折扣：受市场状态影响
        if market_state == "强趋势牛":
            discount = 1.0
        elif market_state == "震荡偏强":
            discount = max(1.0 - max(cv - 0.15, 0) * 0.5, 0.90)
        elif market_state == "震荡偏弱":
            discount = max(1.0 - max(cv - 0.12, 0) * 0.6, 0.85)
        else:
            discount = max(1.0 - max(cv - 0.10, 0) * 0.7, 0.80)
        weighted_score *= discount
        
        final_score = float(np.clip(weighted_score, 0, 100))
        
        # ── 4. 从ReviewManager加载历史胜率（凯利公式用）──
        win_rate = 0.55  # 默认
        avg_win_ratio = 1.5  # 默认盈亏比
        try:
            if os.path.exists(ReviewManager.REVIEW_FILE):
                with open(ReviewManager.REVIEW_FILE, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                total_verified = stats.get("total_verifications", 0)
                overall_acc = stats.get("overall_accuracy_pct", 55) / 100
                if total_verified > 10:
                    win_rate = overall_acc
                    # 从by_date估算盈亏比
                    dates = stats.get("by_date", [])
                    if len(dates) > 3:
                        avg_win_ratio = 1.5
        except:
            pass
        
        # ── 5. 操作建议映射（决策树——无重叠路径）──
        etf_type = etf_info.get("type", "")
        if final_score >= 80:
            if consensus in ("高度一致", "基本一致"):
                operation, holding = "强烈买入", "短期(1-4周)"
            else:
                operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 65:
            operation, holding = "买入", "中期(1-3月)"
        elif final_score >= 50:
            if etf_type == "宽基":
                operation, holding = "长期持有", "长期(6月+)"
            else:
                operation, holding = "持有", "中期(1-3月)"
        elif final_score >= 35:
            operation, holding = "减持", "短期(1-4周)"
        elif final_score >= 20:
            operation, holding = "卖出", "短期(1-4周)"
        else:
            operation, holding = "强烈卖出", "短期(1-4周)"
        
        # 严重分歧修正
        if consensus == "严重分歧":
            if operation in ("强烈买入", "买入"):
                operation, holding = "持有", "中期(1-3月)"
            elif operation in ("长期持有",):
                operation, holding = "减持", "短期(1-4周)"
        
        # ── 6. 凯利公式仓位（受市场状态调节） ──
        market_pos_mult = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}
        pos_mult = market_pos_mult.get(market_state, 1.0)
        adjusted_max_pos = global_max_pos * pos_mult
        kelly_pos = self._kelly_position(win_rate, avg_win_ratio, 1.0, adjusted_max_pos)
        pos_map = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20,
                    "持有": 0.15, "减持": 0.05, "卖出": 0.0, "强烈卖出": 0.0}
        baseline_pos = adjusted_max_pos * pos_map.get(operation, 0.1)
        pos_pct = min(kelly_pos, baseline_pos)
        pos_text_map = {"强烈买入": "重仓", "买入": "中仓", "长期持有": "长持", 
                        "持有": "轻仓", "减持": "减仓", "卖出": "卖出", "强烈卖出": "清仓"}
        
        # ── 7. LLM决策（并行，结果优于规则时覆盖）──
        ratings = [r.rating for r in reports]
        reports_text = "\n\n".join([
            f"【{r.agent_name}】评级:{r.rating} 评分:{r.score} 置信度:{r.confidence}\n分析:{r.analysis[:300]}\n关键因子:{'; '.join(r.key_factors)}\n风险:{'; '.join(r.risk_warnings)}"
            for r in reports
        ])
        debates_text = ""
        if debates:
            for d in debates:
                debates_text += f"\n分歧: {d['topic']}\n"
                for rd in d['rounds']:
                    for k, v in rd.items():
                        if k != 'round':
                            debates_text += f"  {v[:200]}\n"
        
        data_text = f"【ETF信息】{etf_info['name']}({etf_info['code']})\n\n【智能体报告】\n{reports_text}\n\n【辩论记录】\n{debates_text}\n\n【全局仓位上限】{global_max_pos*100:.0f}%"
        llm_out = self._call_llm(self.SYSTEM_PROMPT, 
                                  self._build_user_prompt(etf_info['name'], etf_info['code'], data_text))
        
        if llm_out:
            llm_final_rating = llm_out.get("rating", "中性")
            llm_final_score = float(np.clip(llm_out.get("score", final_score), 0, 100))
            llm_core_logic = llm_out.get("analysis", "")
            # LLM结果优于规则时采用
            if abs(llm_final_score - 50) > abs(final_score - 50):
                final_score = llm_final_score
                final_rating = llm_final_rating
                core_logic = llm_core_logic
            else:
                final_rating = self._score_to_rating(final_score)
                core_logic = f"【规则综合】z-score加权:{final_score:.1f}分 | 共识:{consensus}"
        else:
            final_rating = self._score_to_rating(final_score)
            core_logic = f"【规则综合】z-score加权:{final_score:.1f}分 | Agent数:{len(reports)} | 共识:{consensus}"
        
        all_risks = list(set([w for r in reports for w in r.risk_warnings]))[:5]
        
        # ── 8. 止损止盈计算 ──
        try:
            df = DataCollectAgent.get_etf_price(etf_info['code'])
            hist_vol = float(df["volatility"].rolling(20).mean().iloc[-1])
            vol_factor = max(hist_vol * 100, 1.0)
            stop_loss_pct = round(-max(vol_factor * 2.0, 3.0), 1)
            take_profit_pct = round(max(vol_factor * 4.0, 6.0), 1)
        except:
            stop_loss_pct = -5.0
            take_profit_pct = 15.0
        
        return FinalResearchReport(
            etf_info=etf_info,
            macro_context="",
            agent_reports=reports,
            debates=debates,
            final_rating=final_rating,
            position_suggestion=pos_text_map.get(operation, "观望"),
            suggested_position_pct=round(pos_pct, 2),
            final_score=round(final_score, 1),
            core_logic=core_logic,
            risk_summary="; ".join(all_risks) if all_risks else "暂无显著风险提示",
            consensus_level=consensus,
            operation=operation,
            holding_period=holding,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            factor_contributions=factor_contributions
        )

# ====================== 【完整投研报告输出】 ======================
class ResearchReportGenerator:
    """生成完整的ETF多智能体投研报告——不止表格，含每Agent分析原文+辩论+首席决策"""
    
    @staticmethod
    def generate_full_report(all_reports: list[FinalResearchReport]):
        today = datetime.now().strftime("%Y-%m-%d")
        print("\n" + "█"*160)
        print(f"  ETF 多智能体投研报告 | {today}")
        print("  LLM多角色专家分析 + 矛盾检测 + 选择性辩论 + 首席综合决策")
        print("█"*160)
        
        # 1. 摘要看板
        print("\n" + "="*160)
        print("【📊 投研摘要看板】")
        print("="*160)
        
        summary_rows = []
        for fr in all_reports:
            summary_rows.append({
                "代码": fr.etf_info['code'],
                "名称": fr.etf_info['name'],
                "类型": fr.etf_info['type'],
                "操作建议": fr.operation,
                "持有周期": fr.holding_period,
                "建议仓位": f"{fr.suggested_position_pct*100:.0f}%",
                "共识度": fr.consensus_level,
            })
        
        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))
        
        # 2. 操作建议分层
        strong_buy = [fr for fr in all_reports if fr.operation == "强烈买入"]
        buy = [fr for fr in all_reports if fr.operation == "买入"]
        long_hold = [fr for fr in all_reports if fr.operation == "长期持有"]
        hold = [fr for fr in all_reports if fr.operation == "持有"]
        reduce = [fr for fr in all_reports if fr.operation == "减持"]
        sell = [fr for fr in all_reports if fr.operation in ("卖出", "强烈卖出")]
        
        if strong_buy:
            print(f"\n🔥【强烈买入】（{len(strong_buy)}只）")
            for fr in strong_buy:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        
        if buy:
            print(f"\n📈【买入】（{len(buy)}只）")
            for fr in buy:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        
        if long_hold:
            print(f"\n🏦【长期持有】（{len(long_hold)}只）")
            for fr in long_hold:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 周期:{fr.holding_period} | {fr.core_logic[:80]}")
        
        if hold:
            print(f"\n⏸【持有】（{len(hold)}只）")
            for fr in hold:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 共识:{fr.consensus_level}")
        
        if reduce:
            print(f"\n⬇️【减持】（{len(reduce)}只）")
            for fr in reduce:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        
        if sell:
            print(f"\n🚨【卖出/强烈卖出】（{len(sell)}只）")
            for fr in sell:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        
        # 3. 各标的详细报告
        print("\n" + "="*160)
        print("【📋 各标的多智能体详细报告】")
        print("="*160)
        
        for fr in all_reports:
            ResearchReportGenerator._print_detailed_report(fr)
        
        # 4. 板块平均
        print("\n" + "="*160)
        print("【📈 板块综合评级】")
        print("="*160)
        sector_data = {}
        for fr in all_reports:
            st = fr.etf_info['type']
            sector_data.setdefault(st, []).append(fr.final_rating)
        for sector, ratings_list in sorted(sector_data.items()):
            scores = [RATING_ORDER.index(rt) if rt in RATING_ORDER else 2 for rt in ratings_list]
            avg_idx = np.mean(scores)
            avg_rating = RATING_ORDER[int(round(avg_idx))]
            print(f"  {sector}: {avg_rating}（{len(ratings_list)}只标的）")
        
        # 5. ETF轮动信号
        print("\n" + "="*160)
        print("【📊 ETF轮动信号】")
        print("="*160)
        sorted_by_score = sorted(all_reports, key=lambda x: x.final_score, reverse=True)
        print(f"  🔥 TOP 5 优先买入:")
        for i, fr in enumerate(sorted_by_score[:5], 1):
            op_icon = "🟢" if fr.operation in ("强烈买入", "买入") else "🟡" if fr.operation == "长期持有" else "🔴"
            print(f"    {i}. {fr.etf_info['code']} {fr.etf_info['name']} | 评分:{fr.final_score:.1f} | {op_icon} {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}%")
        print(f"  🧊 BOTTOM 5 建议回避:")
        for i, fr in enumerate(sorted_by_score[-5:], 1):
            op_icon = "🟢" if fr.operation in ("强烈买入", "买入") else "🟡" if fr.operation == "长期持有" else "🔴"
            print(f"    {i}. {fr.etf_info['code']} {fr.etf_info['name']} | 评分:{fr.final_score:.1f} | {op_icon} {fr.operation}")
        
        # 6. 大类资产配置
        print("\n" + "="*160)
        print("【🏛 大类资产配置】")
        print("="*160)
        type_groups = {}
        for fr in all_reports:
            tp = fr.etf_info['type']
            type_groups.setdefault(tp, []).append(fr)
        total_pos = sum(fr.suggested_position_pct for fr in all_reports) or 1
        for tp, group in sorted(type_groups.items()):
            alloc = sum(fr.suggested_position_pct for fr in group) / total_pos * 100
            print(f"  {tp}: {len(group)}只 | 配置占比: {alloc:.1f}%")
            for fr in group:
                print(f"    {fr.etf_info['code']} {fr.etf_info['name']} | {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}%")
        
        # 7. 行业集中度热力图
        print("\n" + "="*160)
        print("【🔥 行业集中度热力图】")
        print("="*160)
        op_counts = {}
        for fr in all_reports:
            op_counts[fr.operation] = op_counts.get(fr.operation, 0) + 1
        total = len(all_reports)
        for op in ["强烈买入", "买入", "长期持有", "持有", "减持", "卖出", "强烈卖出"]:
            cnt = op_counts.get(op, 0)
            if cnt > 0:
                bar = "█" * cnt
                print(f"  {op}: {cnt}只 {bar}")
        buy_ops = {"强烈买入", "买入", "长期持有"}
        sell_ops = {"减持", "卖出", "强烈卖出"}
        buy_cnt = sum(op_counts.get(op, 0) for op in buy_ops)
        sell_cnt = sum(op_counts.get(op, 0) for op in sell_ops)
        buy_ratio = buy_cnt / total * 100
        sell_ratio = sell_cnt / total * 100
        print(f"  多头方向: {buy_cnt}只 ({buy_ratio:.0f}%) | 空头方向: {sell_cnt}只 ({sell_ratio:.0f}%)")
        if buy_ratio > 70:
            print(f"  ⚠️ 集中度预警: 超过{70}%标的集中在多头方向，注意一致性风险")
        if sell_ratio > 40:
            print(f"  ⚠️ 集中度预警: 超过{40}%标的集中在空头方向，市场情绪过度悲观")
        
        # 8. 尾部风险预警
        print("\n" + "="*160)
        print("【⚠️ 尾部风险预警】")
        print("="*160)
        extreme_risk_found = False
        for fr in all_reports:
            all_warnings = []
            for report in fr.agent_reports:
                all_warnings.extend(report.risk_warnings)
            premium_warnings = [w for w in all_warnings if "溢价" in w]
            volume_warnings = [w for w in all_warnings if "流动" in w or "成交量" in w]
            volatility_warnings = [w for w in all_warnings if "波动" in w]
            if premium_warnings:
                extreme_risk_found = True
                print(f"  🔴 溢价风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(premium_warnings[:2])}")
            if volume_warnings:
                extreme_risk_found = True
                print(f"  🟡 流动性风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(volume_warnings[:2])}")
            if volatility_warnings:
                extreme_risk_found = True
                print(f"  🟠 波动风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(volatility_warnings[:2])}")
        if not extreme_risk_found:
            print(f"  ✅ 未检测到尾部风险信号")
        
        # 9. 止损止盈参考
        print("\n" + "="*160)
        print("【🎯 止损止盈参考】")
        print("="*160)
        for fr in all_reports:
            sl = fr.stop_loss_pct
            tp = fr.take_profit_pct
            if sl != 0 or tp != 0:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 止损: {sl:+.1f}% | 止盈: {tp:+.1f}%")
            else:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 止损: 未设置 | 止盈: 未设置")
        
        # 10. 保存台账
        save_path = f"ETF_多智能体投研报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df_summary.to_excel(save_path, index=False)
        print(f"\n✅ 投研摘要已保存：{save_path}")
        
        # 11. 保存完整详细报告到文本文件
        txt_path = f"ETF_多智能体投研报告_{datetime.now().strftime('%Y%m%d')}.txt"
        full_text = ResearchReportGenerator._build_full_report_text(all_reports)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✅ 完整投研报告已保存：{txt_path}")
    
    @staticmethod
    def _build_full_report_text(all_reports: list[FinalResearchReport]) -> str:
        """构建完整投研报告文本（用于保存到文件）"""
        lines = []
        today = datetime.now().strftime("%Y-%m-%d")
        sep = "=" * 160
        lines.append(sep)
        lines.append(f"  ETF 多智能体投研报告 | {today}")
        lines.append("  LLM多角色专家分析 + 矛盾检测 + 选择性辩论 + 首席综合决策")
        lines.append(sep)
        
        # 摘要看板
        lines.append("\n【投研摘要看板】")
        lines.append(sep)
        for fr in all_reports:
            lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} ({fr.etf_info['type']}) | {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}% | 周期:{fr.holding_period} | 共识:{fr.consensus_level}")
        
        # 推荐分层
        strong_buy_text = [fr for fr in all_reports if fr.operation == "强烈买入"]
        buy_text = [fr for fr in all_reports if fr.operation == "买入"]
        long_hold_text = [fr for fr in all_reports if fr.operation == "长期持有"]
        hold_text = [fr for fr in all_reports if fr.operation == "持有"]
        reduce_text = [fr for fr in all_reports if fr.operation == "减持"]
        sell_text = [fr for fr in all_reports if fr.operation in ("卖出", "强烈卖出")]
        if strong_buy_text:
            lines.append(f"\n【强烈买入（{len(strong_buy_text)}只）】")
            for fr in strong_buy_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        if buy_text:
            lines.append(f"\n【买入（{len(buy_text)}只）】")
            for fr in buy_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        if long_hold_text:
            lines.append(f"\n【长期持有（{len(long_hold_text)}只）】")
            for fr in long_hold_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 周期:{fr.holding_period} | {fr.core_logic[:80]}")
        if hold_text:
            lines.append(f"\n【持有（{len(hold_text)}只）】")
            for fr in hold_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 共识:{fr.consensus_level}")
        if reduce_text:
            lines.append(f"\n【减持（{len(reduce_text)}只）】")
            for fr in reduce_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        if sell_text:
            lines.append(f"\n【卖出/强烈卖出（{len(sell_text)}只）】")
            for fr in sell_text:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        
        # 各标的详细报告
        lines.append(f"\n{sep}")
        lines.append("【各标的多智能体详细报告】")
        lines.append(sep)
        
        for fr in all_reports:
            lines.append(f"\n{'─' * 160}")
            lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']}（{fr.etf_info['type']}）")
            lines.append(f"  🎯 操作建议: {fr.operation} | 持有周期: {fr.holding_period}")
            lines.append(f"  仓位: {fr.suggested_position_pct*100:.0f}% | 共识: {fr.consensus_level}")
            lines.append(f"{'─' * 160}")
            
            for report in fr.agent_reports:
                src_tag = "[LLM]" if report.source == "llm" else "[规则]"
                lines.append(f"\n  {src_tag} {report.agent_name}")
                lines.append(f"  评级: {report.rating} ({report.score}分) | 置信度: {report.confidence:.2f}")
                lines.append(f"  {report.analysis[:300]}")
                if report.key_factors:
                    lines.append(f"  关键因子: {'; '.join(report.key_factors)}")
                if report.risk_warnings:
                    lines.append(f"  风险: {'; '.join(report.risk_warnings)}")
            
            if fr.debates:
                lines.append(f"\n  【辩论记录】")
                for d in fr.debates:
                    lines.append(f"  {d['topic']}")
                    for rd in d['rounds']:
                        for k, v in rd.items():
                            if k != 'round':
                                lines.append(f"  {v[:200]}")
            
            lines.append(f"\n  【首席决策】")
            lines.append(f"  最终评级: {fr.final_rating}")
            lines.append(f"  核心逻辑: {fr.core_logic[:200]}")
            lines.append(f"  综合风险: {fr.risk_summary[:200]}")
            if fr.factor_contributions:
                sorted_factors = sorted(fr.factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                contrib_str = " | ".join([f"{name}: {val:+.1f}" for name, val in sorted_factors])
                lines.append(f"  因子贡献: {contrib_str}")
            sl = fr.stop_loss_pct
            tp = fr.take_profit_pct
            if sl != 0 or tp != 0:
                lines.append(f"  止损 {sl:+.1f}% / 止盈 {tp:+.1f}%")
            else:
                lines.append(f"  止损 未设置 / 止盈 未设置")
        
        # 板块评级
        lines.append(f"\n{sep}")
        lines.append("【板块综合评级】")
        lines.append(sep)
        sector_data = {}
        for fr in all_reports:
            st = fr.etf_info['type']
            sector_data.setdefault(st, []).append(fr.final_rating)
        for sector, ratings_list in sorted(sector_data.items()):
            scores = [RATING_ORDER.index(rt) if rt in RATING_ORDER else 2 for rt in ratings_list]
            avg_idx = np.mean(scores)
            avg_rating = RATING_ORDER[int(round(avg_idx))]
            lines.append(f"  {sector}: {avg_rating}（{len(ratings_list)}只标的）")
        
        # ETF轮动信号
        lines.append(f"\n{sep}")
        lines.append("【ETF轮动信号】")
        lines.append(sep)
        sorted_by_score = sorted(all_reports, key=lambda x: x.final_score, reverse=True)
        lines.append("  TOP 5 优先买入:")
        for i, fr in enumerate(sorted_by_score[:5], 1):
            lines.append(f"    {i}. {fr.etf_info['code']} {fr.etf_info['name']} | 评分:{fr.final_score:.1f} | {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}%")
        lines.append("  BOTTOM 5 建议回避:")
        for i, fr in enumerate(sorted_by_score[-5:], 1):
            lines.append(f"    {i}. {fr.etf_info['code']} {fr.etf_info['name']} | 评分:{fr.final_score:.1f} | {fr.operation}")
        
        # 大类资产配置
        lines.append(f"\n{sep}")
        lines.append("【大类资产配置】")
        lines.append(sep)
        type_groups = {}
        for fr in all_reports:
            tp = fr.etf_info['type']
            type_groups.setdefault(tp, []).append(fr)
        total_pos = sum(fr.suggested_position_pct for fr in all_reports) or 1
        for tp, group in sorted(type_groups.items()):
            alloc = sum(fr.suggested_position_pct for fr in group) / total_pos * 100
            lines.append(f"  {tp}: {len(group)}只 | 配置占比: {alloc:.1f}%")
            for fr in group:
                lines.append(f"    {fr.etf_info['code']} {fr.etf_info['name']} | {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}%")
        
        # 行业集中度热力图
        lines.append(f"\n{sep}")
        lines.append("【行业集中度热力图】")
        lines.append(sep)
        op_counts = {}
        for fr in all_reports:
            op_counts[fr.operation] = op_counts.get(fr.operation, 0) + 1
        total = len(all_reports)
        for op in ["强烈买入", "买入", "长期持有", "持有", "减持", "卖出", "强烈卖出"]:
            cnt = op_counts.get(op, 0)
            if cnt > 0:
                lines.append(f"  {op}: {cnt}只")
        buy_ops = {"强烈买入", "买入", "长期持有"}
        sell_ops = {"减持", "卖出", "强烈卖出"}
        buy_cnt = sum(op_counts.get(op, 0) for op in buy_ops)
        sell_cnt = sum(op_counts.get(op, 0) for op in sell_ops)
        buy_ratio = buy_cnt / total * 100
        sell_ratio = sell_cnt / total * 100
        lines.append(f"  多头方向: {buy_cnt}只 ({buy_ratio:.0f}%) | 空头方向: {sell_cnt}只 ({sell_ratio:.0f}%)")
        if buy_ratio > 70:
            lines.append(f"  ⚠️ 集中度预警: 超过{70}%标的集中在多头方向，注意一致性风险")
        if sell_ratio > 40:
            lines.append(f"  ⚠️ 集中度预警: 超过{40}%标的集中在空头方向，市场情绪过度悲观")
        
        # 尾部风险预警
        lines.append(f"\n{sep}")
        lines.append("【尾部风险预警】")
        lines.append(sep)
        extreme_risk_found = False
        for fr in all_reports:
            all_warnings = []
            for report in fr.agent_reports:
                all_warnings.extend(report.risk_warnings)
            premium_warnings = [w for w in all_warnings if "溢价" in w]
            volume_warnings = [w for w in all_warnings if "流动" in w or "成交量" in w]
            volatility_warnings = [w for w in all_warnings if "波动" in w]
            if premium_warnings:
                extreme_risk_found = True
                lines.append(f"  溢价风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(premium_warnings[:2])}")
            if volume_warnings:
                extreme_risk_found = True
                lines.append(f"  流动性风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(volume_warnings[:2])}")
            if volatility_warnings:
                extreme_risk_found = True
                lines.append(f"  波动风险: {fr.etf_info['code']} {fr.etf_info['name']} | {'; '.join(volatility_warnings[:2])}")
        if not extreme_risk_found:
            lines.append("  未检测到尾部风险信号")
        
        # 止损止盈参考
        lines.append(f"\n{sep}")
        lines.append("【止损止盈参考】")
        lines.append(sep)
        for fr in all_reports:
            sl = fr.stop_loss_pct
            tp = fr.take_profit_pct
            if sl != 0 or tp != 0:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 止损: {sl:+.1f}% | 止盈: {tp:+.1f}%")
            else:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 止损: 未设置 | 止盈: 未设置")
        
        return "\n".join(lines)

    @staticmethod
    def _print_detailed_report(fr: FinalResearchReport):
        """打印单只ETF的详细投研报告"""
        print(f"\n{'─'*160}")
        print(f"  {fr.etf_info['code']} {fr.etf_info['name']}（{fr.etf_info['type']}）")
        print(f"  🎯 操作建议: {fr.operation} | 持有周期: {fr.holding_period}")
        print(f"  仓位: {fr.suggested_position_pct*100:.0f}% | 共识: {fr.consensus_level}")
        print(f"{'─'*160}")
        
        for report in fr.agent_reports:
            src_tag = "🤖LLM" if report.source == "llm" else "⚙️规则"
            print(f"\n  ┌─ {src_tag} {report.agent_name} ────────────────────────")
            print(f"  │ 评级: {report.rating} ({report.score}分) | 置信度: {report.confidence:.2f}")
            print(f"  │ {report.analysis[:300]}")
            if report.key_factors:
                print(f"  │ 关键因子: {'; '.join(report.key_factors)}")
            if report.risk_warnings:
                print(f"  │ ⚠️ 风险: {'; '.join(report.risk_warnings)}")
            print(f"  └────────────────────────────────────────────")
        
        if fr.debates:
            print(f"\n  ⚔️ 【辩论记录】")
            for d in fr.debates:
                print(f"    🎯 {d['topic']}")
                for rd in d['rounds']:
                    for k, v in rd.items():
                        if k != 'round':
                            print(f"    {v[:200]}")
        
        print(f"\n  📌 【首席决策】")
        print(f"  最终评级: {fr.final_rating}")
        print(f"  核心逻辑: {fr.core_logic[:200]}")
        print(f"  综合风险: {fr.risk_summary[:200]}")
        if fr.factor_contributions:
            sorted_factors = sorted(fr.factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            contrib_str = " | ".join([f"{name}: {val:+.1f}" for name, val in sorted_factors])
            print(f"  因子贡献: {contrib_str}")
        sl = fr.stop_loss_pct
        tp = fr.take_profit_pct
        if sl != 0 or tp != 0:
            print(f"  止损 {sl:+.1f}% / 止盈 {tp:+.1f}%")
        else:
            print(f"  止损 未设置 / 止盈 未设置")

# ====================== 【8. 单标的并行任务入口（旧版-保留供参考）】 ======================
# 注意：此函数引用了旧的 DecisionAgent，仅保留供参考
# 新版使用 MainSchedulerAgent._research_single_etf 替代
def single_etf_all_agents(etf_item: dict, global_max_pos: float) -> dict:
    code = etf_item["code"]
    name = etf_item["name"]
    typ = etf_item["type"]
    idx = etf_item["index_code"]

    with ThreadPoolExecutor(max_workers=AGENT_WORKERS) as agent_exec:
        f_val = agent_exec.submit(ValueScoreAgent.run, idx)
        f_boom = agent_exec.submit(BoomScoreAgent.run, code)
        f_tech = agent_exec.submit(TechScoreAgent.run, code)
        f_fund = agent_exec.submit(FundScoreAgent.run, code, idx)
        f_risk = agent_exec.submit(RiskScoreAgent.run, code)
        f_opinion = agent_exec.submit(PublicOpinionAgent.run, name, code)
        f_bt = agent_exec.submit(BacktestAgent.run, code)
        f_premium = agent_exec.submit(DataCollectAgent.get_etf_premium, code)

        s_val = f_val.result()
        s_boom = f_boom.result()
        s_tech = f_tech.result()
        s_fund = f_fund.result()
        s_risk = f_risk.result()
        opinion_res = f_opinion.result()
        bt_data = f_bt.result()
        premium = f_premium.result()

    s_opinion = opinion_res["opinion_score"]
    op_tag = opinion_res["opinion_tag"]
    op_desc = opinion_res["opinion_desc"]
    news = opinion_res["news_content"]
    kw = opinion_res["keywords"]
    trend = opinion_res["opinion_trend"]
    llm_text = opinion_res["llm_explain"]
    warn_flag = opinion_res["warn_alert"]

    total_score, pos_rate, action = DecisionAgent.run(
        global_max_pos, s_val, s_boom, s_tech, s_fund, s_risk, s_opinion, warn_flag
    )

    return {
        "代码": code,
        "名称": name,
        "板块类型": typ,
        "估值分": s_val,
        "景气动量分": s_boom,
        "技术趋势分": s_tech,
        "资金流分": s_fund,
        "风控安全分": s_risk,
        "舆情分数": s_opinion,
        "舆情标签": op_tag,
        "舆情简述": op_desc,
        "舆情趋势": trend,
        "资讯摘要": news,
        "情绪关键词": kw,
        "大模型解读": llm_text,
        "利空告警": warn_flag,
        "综合总分": total_score,
        "折溢价率(%)": round(premium*100,2),
        "历史胜率(%)": bt_data["胜率"],
        "盈亏比": bt_data["盈亏比"],
        "最大回撤(%)": bt_data["最大回撤"],
        "120日收益(%)": bt_data["回测收益"],
        "建议仓位": f"{pos_rate*100:.0f}%",
        "操作建议": action
    }

# ====================== 【复盘引擎】 ======================
class ReviewManager:
    """复盘引擎：保存每日快照、T+1验证操作建议、跟踪累计准确率"""
    
    SNAPSHOT_DIR = "data/snapshots"
    REVIEW_FILE = "data/review/cumulative_stats.json"
    
    @classmethod
    def process(cls, all_reports: list[FinalResearchReport]):
        """执行完整复盘流程：保存今日快照 → T+1验证 → 更新累计统计 → 打印"""
        os.makedirs(f"{cls.SNAPSHOT_DIR}", exist_ok=True)
        os.makedirs("data/review", exist_ok=True)
        
        # 1. 保存今日快照
        cls._save_snapshot(all_reports)
        
        # 2. T+1验证昨日建议
        verifications = cls._run_t1_verification()
        if not verifications:
            return  # 首次运行，无历史数据
        
        # 3. 更新累计统计
        cls._update_cumulative_stats(verifications)
        
        # 4. 打印复盘摘要
        cls._print_review_summary()

        # 4b. 校准曲线
        cls._print_calibration_curve()

        # 5. 统计显著性检验
        cls._compute_significance()

        # 6. 数据新鲜度校验
        cls._check_data_freshness()

        # 7. 更新Agent权重文件
        cls._update_agent_weights()
    
    @classmethod
    def _save_snapshot(cls, all_reports: list[FinalResearchReport]):
        """保存今日操作建议快照"""
        today = datetime.now().strftime("%Y%m%d")
        etfs = []
        for fr in all_reports:
            agents_data = [{"n": r.agent_name, "rt": r.rating, "sc": r.score, "src": r.source[:4]}
                          for r in fr.agent_reports]
            # 从缓存获取收盘价
            try:
                df = DataCollectAgent.get_etf_price(fr.etf_info['code'])
                close_px = float(df["close"].iloc[-1])
            except Exception:
                close_px = 0.0
            
            etfs.append({
                "code": fr.etf_info['code'], "name": fr.etf_info['name'], "type": fr.etf_info['type'],
                "operation": fr.operation, "holding_period": fr.holding_period,
                "final_score": fr.final_score, "final_rating": fr.final_rating,
                "position_pct": fr.suggested_position_pct, "consensus": fr.consensus_level,
                "close_price": close_px,
                "agents": agents_data
            })
        
        snapshot = {
            "date": today,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_volume_bn": DataCollectAgent.get_market_total_volume(),
            "etfs": etfs
        }
        path = f"{cls.SNAPSHOT_DIR}/{today}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def _get_prev_snapshot(cls) -> tuple[dict, str] | None:
        """获取最近一次历史快照"""
        files = sorted(glob.glob(f"{cls.SNAPSHOT_DIR}/*.json"))
        if len(files) < 2:
            return None  # 只有今日文件或没有
        prev_path = files[-2]  # 上一次运行的快照
        prev_date = os.path.basename(prev_path).replace(".json", "")
        with open(prev_path, "r", encoding="utf-8") as f:
            return json.load(f), prev_date
    
    @classmethod
    def _run_t1_verification(cls) -> list[dict]:
        """T+1验证：对比昨日操作建议与今日实际涨跌"""
        result = cls._get_prev_snapshot()
        if result is None:
            return []
        prev_data, prev_date = result
        
        verifications = []
        for etf in prev_data["etfs"]:
            code = etf["code"]
            prev_price = etf["close_price"]
            if prev_price == 0:
                continue
            
            # 获取今日实际价格（从缓存）
            try:
                df = DataCollectAgent.get_etf_price(code)
                curr_price = float(df["close"].iloc[-1])
            except Exception:
                continue
            
            ret_pct = (curr_price - prev_price) / prev_price * 100
            is_correct = cls._check_direction(etf["operation"], ret_pct)
            
            verifications.append({
                "date": prev_date, "code": code, "name": etf["name"],
                "operation": etf["operation"], "return_pct": round(ret_pct, 2),
                "prev_price": prev_price, "curr_price": curr_price,
                "correct": is_correct
            })
        
        return verifications
    
    @staticmethod
    def _check_direction(operation: str, ret_pct: float) -> str:
        """判断操作建议方向是否正确"""
        buy_ops = {"强烈买入", "买入"}
        sell_ops = {"卖出", "强烈卖出"}
        hold_ops = {"长期持有", "持有"}
        
        if operation in buy_ops:
            if ret_pct > 0.5: return "正确"
            if ret_pct < -0.5: return "错误"
            return "持平"
        elif operation in sell_ops:
            if ret_pct < -0.5: return "正确"
            if ret_pct > 0.5: return "错误"
            return "持平"
        elif operation == "减持":
            if ret_pct < -0.3: return "正确"
            if ret_pct > 0.5: return "错误"
            return "持平"
        else:  # 持有/长期持有
            if -1.0 <= ret_pct <= 1.0: return "正确"
            if ret_pct < -1.0: return "错误"
            return "正确"  # 上涨也算对
    
    @classmethod
    def _update_cumulative_stats(cls, verifications: list[dict]):
        """更新累计复盘统计"""
        stats = {"last_date": "", "total_runs": 0, "total_verifications": 0,
                 "overall_accuracy_pct": 0.0, "by_operation": {}, "by_agent": {},
                 "by_date": [], "pending": []}
        
        # 加载已有统计
        if os.path.exists(cls.REVIEW_FILE):
            with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        
        today = datetime.now().strftime("%Y%m%d")
        
        # 统计本次验证结果
        correct = sum(1 for v in verifications if v["correct"] == "正确")
        wrong = sum(1 for v in verifications if v["correct"] == "错误")
        
        # 更新按操作类型
        for v in verifications:
            op = v["operation"]
            if op not in stats["by_operation"]:
                stats["by_operation"][op] = {"total": 0, "correct": 0, "wrong": 0}
            stats["by_operation"][op]["total"] += 1
            if v["correct"] == "正确":
                stats["by_operation"][op]["correct"] += 1
            elif v["correct"] == "错误":
                stats["by_operation"][op]["wrong"] += 1
        
        # 更新日期记录
        stats["by_date"].append({
            "d": verifications[0]["date"] if verifications else today,
            "verified": len(verifications),
            "correct": correct,
            "wrong": wrong,
            "pct": round(correct / (correct + wrong) * 100, 1) if (correct + wrong) > 0 else 0
        })
        
        # 更新Agent维度准确率（从最近快照读取agent评分方向 vs 实际涨跌）
        cls._update_agent_stats(stats)
        
        # 重新计算总体
        total_correct = sum(d["correct"] for d in stats["by_date"])
        total_wrong = sum(d["wrong"] for d in stats["by_date"])
        total_v = total_correct + total_wrong
        stats["overall_accuracy_pct"] = round(total_correct / total_v * 100, 1) if total_v > 0 else 0
        stats["total_verifications"] = total_v
        stats["total_runs"] = len(stats["by_date"])
        stats["last_date"] = today
        
        with open(cls.REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def _update_agent_stats(cls, stats: dict):
        """更新各智能体的方向准确率"""
        result = cls._get_prev_snapshot()
        if result is None:
            return
        prev_data, prev_date = result
        
        today_prices = {}  # 缓存今日价格
        for etf in prev_data["etfs"]:
            code = etf["code"]
            if code not in today_prices:
                try:
                    df = DataCollectAgent.get_etf_price(code)
                    today_prices[code] = float(df["close"].iloc[-1])
                except Exception:
                    continue
            
            prev_price = etf["close_price"]
            curr_price = today_prices[code]
            if prev_price == 0:
                continue
            actual_ret = (curr_price - prev_price) / prev_price * 100
            actual_dir = 1 if actual_ret > 0.3 else (-1 if actual_ret < -0.3 else 0)
            
            for a in etf["agents"]:
                aname = a["n"]
                if aname not in stats["by_agent"]:
                    stats["by_agent"][aname] = {"directional": 0, "hits": 0}
                
                rating_dir = cls._agent_direction(a["rt"])
                if rating_dir == 0 or actual_dir == 0:
                    continue  # 中性不参与方向统计
                
                stats["by_agent"][aname]["directional"] += 1
                if rating_dir == actual_dir:
                    stats["by_agent"][aname]["hits"] += 1
    
    @staticmethod
    def _agent_direction(rating: str) -> int:
        """将评级转为方向信号"""
        if rating in ("强烈看多", "看多"): return 1
        if rating in ("强烈看空", "看空"): return -1
        return 0
    
    @classmethod
    def _print_review_summary(cls):
        """打印复盘摘要到控制台"""
        if not os.path.exists(cls.REVIEW_FILE):
            return
        with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        print("\n" + "="*160)
        print("【📊 昨日复盘 | T+1验证】")
        print("="*160)
        
        if stats["by_date"]:
            last = stats["by_date"][-1]
            print(f"  验证日期: {last['d']} → {datetime.now().strftime('%Y%m%d')}")
            print(f"  验证标的: {last['verified']}只 | 正确: {last['correct']} | 错误: {last['wrong']} | 准确率: {last['pct']}%")
        
        print(f"\n  📈 累计复盘({stats['total_runs']}天/{stats['total_verifications']}次)")
        print(f"  整体准确率: {stats['overall_accuracy_pct']}%")
        
        # 按操作类型
        if stats["by_operation"]:
            print(f"\n  按操作类型:")
            for op, data in sorted(stats["by_operation"].items()):
                hit = data["correct"]
                tot = data["total"]
                p = round(hit / tot * 100, 1) if tot > 0 else 0
                bar = "█" * int(p / 10) + "░" * (10 - int(p / 10))
                print(f"    {op}: {bar} {p}% ({hit}/{tot})")
        
        # 按Agent（排序）
        if stats["by_agent"]:
            print(f"\n  智能体方向准确率排行:")
            sorted_agents = sorted(stats["by_agent"].items(), key=lambda x: x[1]["hits"]/max(x[1]["directional"],1), reverse=True)
            for rank, (name, data) in enumerate(sorted_agents, 1):
                d = data["directional"]
                h = data["hits"]
                p = round(h / d * 100, 1) if d > 0 else 0
                bar = "█" * int(p / 10) + "░" * (10 - int(p / 10))
                star = " ★" if rank == 1 else ""
                print(f"    {rank}. {name}: {bar} {p}% ({h}/{d}){star}")
        
        print()

    @classmethod
    def _print_calibration_curve(cls):
        """校准曲线：将评分分为十档，计算每档的实际平均收益"""
        files = sorted(glob.glob(f"{cls.SNAPSHOT_DIR}/*.json"))
        if len(files) < 2:
            return

        buckets = {i: {"returns": [], "count": 0} for i in range(0, 100, 10)}

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                snap = json.load(f)
            for etf in snap.get("etfs", []):
                score = etf.get("final_score", 50)
                px = etf.get("close_price", 0)
                if px == 0:
                    continue
                try:
                    df = DataCollectAgent.get_etf_price(etf["code"])
                except Exception:
                    continue
                sd = snap["date"]
                dl = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").tolist()
                if sd not in dl:
                    continue
                i = dl.index(sd)
                if i + 1 >= len(df):
                    continue
                ret = df.iloc[i + 1]["close"] / px - 1
                bucket = (int(score) // 10) * 10
                buckets[bucket]["returns"].append(ret)
                buckets[bucket]["count"] += 1

        total = sum(d["count"] for d in buckets.values())
        if total < 2:
            return

        all_rets = [r for d in buckets.values() for r in d["returns"]]
        overall = float(np.mean(all_rets)) * 100

        print("\n" + "="*80)
        print("【📊 校准曲线 | Calibration Curve】")
        print("="*80)
        print(f"  {'评分区间':<12} {'数量':<6} {'平均收益%':<10} {'校准偏差':<10}")
        print(f"  {'-'*38}")

        for bucket in range(0, 100, 10):
            d = buckets[bucket]
            if d["count"] == 0:
                continue
            avg = float(np.mean(d["returns"])) * 100
            bias = avg - overall
            print(f"  {bucket:3d}-{bucket+9:<6} {d['count']:<6} {avg:<+8.2f}%  {bias:<+8.2f}%")

        print(f"\n  总体均值: {overall:+.2f}%")
        print()

    @classmethod
    def _run_rolling_backtest(cls) -> list[dict]:
        """滚动回测：针对120+数据点的ETF，模拟20日滚动窗口方向准确率"""
        print("\n【滚动回测】检验趋势信号历史表现...")
        results = []
        pool = MainSchedulerAgent.ETF_POOL
        for item in pool:
            code = item["code"]
            try:
                df = DataCollectAgent.get_etf_price(code)
                if df is None or len(df) < 120:
                    continue
                window_size = 20
                correct, total = 0, 0
                returns = []
                for i in range(len(df) - window_size - 1):
                    window = df.iloc[i:i + window_size]
                    curr_close = df.iloc[i + window_size]["close"]
                    next_close = df.iloc[i + window_size + 1]["close"]
                    ma5 = window["close"].tail(5).mean()
                    ma20 = window["close"].tail(20).mean()
                    signal = 1 if ma5 > ma20 else -1
                    actual_ret = (next_close - curr_close) / curr_close
                    actual_dir = 1 if actual_ret > 0 else -1
                    if signal == actual_dir:
                        correct += 1
                    total += 1
                    returns.append(actual_ret)
                if total == 0:
                    continue
                win_rate = correct / total * 100
                avg_ret = float(np.mean(returns)) * 100
                std_ret = float(np.std(returns)) * 100
                sharpe = (avg_ret / max(std_ret, 1e-6)) * (252 ** 0.5)
                results.append({
                    "code": code, "name": item["name"],
                    "windows": total, "win_rate": round(win_rate, 1),
                    "sharpe": round(sharpe, 2)
                })
            except Exception:
                continue
        if results:
            avg_win = float(np.mean([r["win_rate"] for r in results]))
            avg_sharpe = float(np.mean([r["sharpe"] for r in results]))
            print(f"  回测ETF: {len(results)}只 | 平均胜率: {avg_win:.1f}% | 平均夏普: {avg_sharpe:.2f}")
            for r in sorted(results, key=lambda x: x["win_rate"], reverse=True)[:5]:
                print(f"    {r['code']} {r['name']}: 胜率{r['win_rate']}% 夏普{r['sharpe']} (窗口{r['windows']})")
        else:
            print("  无满足条件的ETF（需120+数据点）")
        return results

    @classmethod
    def _compute_significance(cls):
        """二项检验：验证整体准确率是否显著高于随机(50%)"""
        if not os.path.exists(cls.REVIEW_FILE):
            return
        with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
        total = stats.get("total_verifications", 0)
        correct = sum(d["correct"] for d in stats.get("by_date", []))
        if total < 3:
            return
        p_null = 0.5
        if correct > total / 2:
            tail_sum = 0.0
            for k in range(correct, total + 1):
                log_p = log(comb(total, k)) + k * log(p_null) + (total - k) * log(1 - p_null)
                tail_sum += exp(log_p)
            p_value = min(tail_sum * 2, 1.0)
        else:
            tail_sum = 0.0
            for k in range(0, correct + 1):
                log_p = log(comb(total, k)) + k * log(p_null) + (total - k) * log(1 - p_null)
                tail_sum += exp(log_p)
            p_value = min(tail_sum * 2, 1.0)
        sig = "✅ 统计显著(p<0.05)" if p_value < 0.05 else "⚠️ 未达统计显著"
        print(f"\n【统计检验】二项检验: n={total} 正确={correct} p值={p_value:.4f} | {sig}")

    @classmethod
    def _check_data_freshness(cls):
        """校验今日ETF数据新鲜度"""
        print("\n【数据新鲜度校验】")
        pool = MainSchedulerAgent.ETF_POOL[:5]
        today = datetime.now()
        stale, ok = 0, 0
        for item in pool:
            code = item["code"]
            try:
                df = DataCollectAgent.get_etf_price(code)
                if df is None or df.empty or "date" not in df.columns:
                    print(f"  ⚠️ {code} {item['name']}: 数据为空")
                    stale += 1
                    continue
                last_date = pd.to_datetime(df["date"].iloc[-1])
                days_diff = (today - last_date).days
                if days_diff > 5:
                    print(f"  ⚠️ {code} {item['name']}: 数据陈旧({days_diff}天前 {last_date.strftime('%Y-%m-%d')})")
                    stale += 1
                else:
                    print(f"  ✅ {code} {item['name']}: 最新{last_date.strftime('%Y-%m-%d')}({days_diff}天前)")
                    ok += 1
            except Exception as e:
                print(f"  ❌ {code} {item['name']}: 获取异常 → {str(e)[:60]}")
                stale += 1
        total_checked = ok + stale
        if total_checked > 0 and stale > total_checked / 2:
            print(f"  ⚠️ 警告: {stale}/{total_checked} 数据异常，结果可能不准确")
        else:
            print(f"  数据新鲜度: {ok}/{total_checked} 正常")

    @classmethod
    def _update_agent_weights(cls):
        """持久化Agent权重数据供ChiefDecisionAgent.load_agent_weights()读取"""
        stats = {"last_date": "", "total_runs": 0, "total_verifications": 0,
                 "overall_accuracy_pct": 0.0, "by_operation": {}, "by_agent": {},
                 "by_date": [], "pending": []}
        if os.path.exists(cls.REVIEW_FILE):
            with open(cls.REVIEW_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        if not stats.get("by_agent"):
            cls._update_agent_stats(stats)
        with open(cls.REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


# ====================== 【顶层主控调度 - 三段式多智能体】 ======================
class MainSchedulerAgent:
    # ETF标的池
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

    @staticmethod
    def _apply_correlation_constraint(final_reports: list[FinalResearchReport]):
        codes = [fr.etf_info['code'] for fr in final_reports]
        price_data = {}
        for code in codes:
            try:
                df = DataCollectAgent.get_etf_price(code)
                if len(df) >= 60:
                    price_data[code] = df["close"].tail(60).values
            except Exception:
                pass
        if len(price_data) < 2:
            return
        sorted_codes = list(price_data.keys())
        price_matrix = np.array([price_data[c] for c in sorted_codes])
        corr_matrix = np.corrcoef(price_matrix)
        n = len(sorted_codes)
        visited = set()
        report_map = {fr.etf_info['code']: fr for fr in final_reports}
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
                group_codes = [sorted_codes[idx] for idx in group]
                group_reports = [report_map[c] for c in group_codes if c in report_map]
                if not group_reports:
                    continue
                total_pos = sum(fr.suggested_position_pct for fr in group_reports)
                if total_pos > 0.3:
                    scale = 0.3 / total_pos
                    for fr in group_reports:
                        fr.suggested_position_pct = round(fr.suggested_position_pct * scale, 4)
                    names = [fr.etf_info['name'] for fr in group_reports]
                    print(f"  🔗 相关性约束: {' ↔ '.join(names)} 高度相关(>{0.8}), 总仓位{total_pos*100:.0f}% → 30%")
    
    def __init__(self):
        self.macro_agent = MacroAnalystAgent()
        self.monetary_agent = MonetaryPolicyAgent()
        self.policy_agent = PolicyEventAgent()
        self.value_agent = ValueAnalystAgent()
        self.tech_agent = TechAnalystAgent()
        self.sentiment_agent = SentimentAnalystAgent()
        self.fundflow_agent = FundFlowAnalystAgent()
        self.risk_agent = RiskManagerAgent()
        self.industry_agent = IndustryAnalystAgent()
        self.retail_sentiment_agent = RetailSentimentAgent()
        self.cross_market_agent = CrossMarketAgent()
        self.chief_agent = ChiefDecisionAgent()
        self.market_state = "震荡偏强"

    def detect_market_state(self) -> str:
        try:
            volume = DataCollectAgent.get_market_total_volume()

            df_csi = DataCollectAgent.get_etf_price("510300")
            if len(df_csi) < 20:
                return "震荡偏强"
            csi_close = df_csi["close"].iloc[-1]
            csi_ma5 = df_csi["ma5"].iloc[-1]
            csi_ma20 = df_csi["ma20"].iloc[-1]
            if csi_close > csi_ma5 and csi_ma5 > csi_ma20:
                csi_trend = 1
            elif csi_close < csi_ma5 and csi_ma5 < csi_ma20:
                csi_trend = -1
            else:
                csi_trend = 0

            csi_vol = df_csi["volatility"].tail(20).mean()
            vol_high = csi_vol > 0.025

            above_ma20 = 0
            total_checked = 0
            for item in self.ETF_POOL:
                try:
                    df = DataCollectAgent.get_etf_price(item["code"])
                    if len(df) >= 20 and df["close"].iloc[-1] > df["ma20"].iloc[-1]:
                        above_ma20 += 1
                    total_checked += 1
                except Exception:
                    pass
            breadth = above_ma20 / max(total_checked, 1)

            if volume >= 10000 and csi_trend > 0 and breadth > 0.6:
                state = "强趋势牛"
            elif csi_trend < 0 and breadth < 0.3 and volume < 7000:
                state = "强趋势熊"
            elif volume >= 7000 and csi_trend >= 0 and breadth > 0.4:
                state = "震荡偏强"
            elif volume < 7000 or (csi_trend < 0 and breadth < 0.4):
                state = "震荡偏弱"
            else:
                state = "震荡偏强"

            if vol_high and state in ("震荡偏强",):
                state = "震荡偏弱"
            return state
        except Exception:
            return "震荡偏强"

    def _rank_etf_tiers(self) -> dict[int, list]:
        wide = [item for item in self.ETF_POOL if item["type"] == "宽基"]
        others = [item for item in self.ETF_POOL if item["type"] != "宽基"]
        scored = []
        for item in others:
            try:
                df = DataCollectAgent.get_etf_price(item["code"])
                if len(df) < 20:
                    scored.append({**item, "_comp": 0})
                    continue
                c = df["close"].values[-20:]
                v = df["volume"].values[-20:]
                vola = df["volatility"].values[-20:]
                scored.append({**item, "_volume": float(np.mean(v)),
                               "_volatility": float(np.mean(vola)),
                               "_momentum_abs": float(abs(c[-1] / c[0] - 1))})
            except Exception:
                scored.append({**item, "_comp": 0})
        valid = [s for s in scored if "_comp" not in s]
        n = len(valid)
        if n == 0:
            return {1: list(wide), 2: [], 3: []}
        vols = np.array([s["_volume"] for s in valid])
        volas = np.array([s["_volatility"] for s in valid])
        moms = np.array([s["_momentum_abs"] for s in valid])
        r_vol = pd.Series(vols).rank(method="average").values / n
        r_vola = pd.Series(volas).rank(method="average").values / n
        r_mom = pd.Series(moms).rank(method="average").values / n
        for i, s in enumerate(valid):
            s["_comp"] = r_vol[i] * 0.4 + r_vola[i] * 0.3 + r_mom[i] * 0.3
        valid.sort(key=lambda x: x["_comp"], reverse=True)
        clean = [{"code": s["code"], "name": s["name"], "type": s["type"], "index_code": s["index_code"]} for s in valid]
        return {1: list(wide) + clean[:7], 2: clean[7:17], 3: clean[17:]}

    def _rule_based_research(self, code: str, name: str, typ: str, idx: str,
                              global_max_pos: float, macro_report: AgentReport,
                              market_state: str) -> FinalResearchReport:
        rs_value = ValueScoreAgent.run(idx)
        rs_tech = TechScoreAgent.run(code)
        rs_fund = FundScoreAgent.run(code, idx)
        rs_risk = RiskScoreAgent.run(code)
        reports = [
            self.chief_agent._parse_to_report(code, name, None, rs_value),
            self.chief_agent._parse_to_report(code, name, None, rs_tech),
            self.chief_agent._parse_to_report(code, name, None, rs_fund),
            self.chief_agent._parse_to_report(code, name, None, rs_risk),
        ]
        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
        adjusted = []
        for r in reports:
            s = r.score
            if r.agent_name in ChiefDecisionAgent.REVERSE_AGENTS:
                s = 100 - s
            adjusted.append(s)
        zs = ChiefDecisionAgent._compute_zscore(adjusted)
        ns = [float(np.clip(50 + z * 15, 0, 100)) for z in zs]
        wm = ChiefDecisionAgent.load_agent_weights()
        wv = []
        for r in reports:
            w = wm.get(r.agent_name, 1.0) * (0.5 + r.confidence)
            wv.append(w)
        tw = sum(wv)
        ws_score = sum((w / tw) * s for w, s in zip(wv, ns)) if tw > 0 else float(np.mean(ns))
        ts_mom = ChiefDecisionAgent._time_series_momentum_score(code)
        fs = float(np.clip(ws_score + ts_mom, 0, 100))
        cv = float(np.std(ns)) / max(float(np.mean(ns)), 1)
        consensus = "高度一致" if cv < 0.1 else "基本一致" if cv < 0.2 else "存在分歧" if cv < 0.35 else "严重分歧"
        fr = BaseLLMAgent._score_to_rating(fs)
        pm = {"强趋势牛": 1.2, "震荡偏强": 1.0, "震荡偏弱": 0.8, "强趋势熊": 0.5}.get(market_state, 1.0)
        am = global_max_pos * pm
        kp = ChiefDecisionAgent._kelly_position(0.55, 1.5, 1.0, am)
        if fs >= 80:
            op, hd = ("强烈买入", "短期(1-4周)") if consensus in ("高度一致", "基本一致") else ("买入", "中期(1-3月)")
        elif fs >= 65:
            op, hd = "买入", "中期(1-3月)"
        elif fs >= 50:
            op, hd = ("长期持有", "长期(6月+)") if typ == "宽基" else ("持有", "中期(1-3月)")
        elif fs >= 35:
            op, hd = "减持", "短期(1-4周)"
        elif fs >= 20:
            op, hd = "卖出", "短期(1-4周)"
        else:
            op, hd = "强烈卖出", "短期(1-4周)"
        if consensus == "严重分歧":
            if op in ("强烈买入", "买入"):
                op, hd = "持有", "中期(1-3月)"
            elif op in ("长期持有",):
                op, hd = "减持", "短期(1-4周)"
        po = {"强烈买入": 0.35, "买入": 0.25, "长期持有": 0.20, "持有": 0.15, "减持": 0.05, "卖出": 0.0, "强烈卖出": 0.0}
        pp = min(kp, am * po.get(op, 0.1))
        pt = {"强烈买入": "重仓", "买入": "中仓", "长期持有": "长持", "持有": "轻仓", "减持": "减仓", "卖出": "卖出", "强烈卖出": "清仓"}
        ar = list(set([w for r in reports for w in r.risk_warnings]))[:5]
        try:
            dr = DataCollectAgent.get_etf_price(code)
            hv = float(dr["volatility"].rolling(20).mean().iloc[-1])
            vf = max(hv * 100, 1.0)
            sl, tp = round(-max(vf * 2.0, 3.0), 1), round(max(vf * 4.0, 6.0), 1)
        except:
            sl, tp = -5.0, 15.0
        return FinalResearchReport(
            etf_info={"code": code, "name": name, "type": typ, "index_code": idx},
            macro_context=macro_report.analysis[:200], agent_reports=reports, debates=[],
            final_rating=fr, position_suggestion=pt.get(op, "观望"),
            suggested_position_pct=round(pp, 2), final_score=round(fs, 1),
            core_logic=f"【规则评分】z-score加权:{ws_score:.1f}分 | 动量:{ts_mom:+.1f} | 共识:{consensus}",
            risk_summary="; ".join(ar) if ar else "暂无显著风险提示",
            consensus_level=consensus, operation=op, holding_period=hd,
            stop_loss_pct=sl, take_profit_pct=tp, factor_contributions={}
        )

    def run(self):
        print("█"*160)
        print("  ETF 多智能体投研系统 | LLM多角色专家分析 + 辩论 + 首席决策")
        print("█"*160)

        # Phase 0: 宏观分析
        print(f"\n【Phase 0】宏观环境分析...")
        macro_report = self.macro_agent.run()
        print(f"  ✅ 宏观: {macro_report.rating} ({macro_report.score}分) | {macro_report.analysis[:80]}")
        global_max_pos = macro_report.score / 100

        if not LLM_ENABLED:
            print("  ⚙️ LLM开关=OFF，使用规则评分模式")

        # Phase 0.5: 全局政策&流动性分析
        print(f"\n【Phase 0.5】政策与流动性分析...")
        monetary_report = self.monetary_agent.run()
        print(f"  ✅ 货币政策: {monetary_report.rating} ({monetary_report.score}分)")
        policy_report = self.policy_agent.run()
        print(f"  ✅ 政策周期: {policy_report.rating} ({policy_report.score}分)")

        print(f"\n【全局仓位上限】{global_max_pos*100:.0f}%")
        print(f"【外层并发】{MAIN_WORKERS} | 【Agent并发】{AGENT_WORKERS}\n")

        # Phase 0.75: 市场状态检测
        self.market_state = self.detect_market_state()
        print(f"【市场状态】{self.market_state}")

        # Phase 0.6: ETF分层
        tiers = self._rank_etf_tiers()
        print(f"\n【分析深度分层】")
        print(f"  Tier 1 (完整 8-Agent+辩论): {len(tiers[1])} 只 — {' '.join(t['name'] for t in tiers[1])}")
        print(f"  Tier 2 (简化 4-Agent 无辩论): {len(tiers[2])} 只")
        print(f"  Tier 3 (纯规则评分): {len(tiers[3])} 只")
        print()

        # Phase 1: 按Tier顺序处理
        final_reports = []
        tier_labels = {1: "完整分析", 2: "简化分析", 3: "规则评分"}
        for tn in [1, 2, 3]:
            pool = tiers[tn]
            if not pool:
                continue
            print(f"【Phase 1 - Tier{tn} {tier_labels[tn]}】{len(pool)}只ETF...")
            with ThreadPoolExecutor(max_workers=MAIN_WORKERS) as exec:
                fmap = {
                    exec.submit(self._research_single_etf, item, global_max_pos,
                                macro_report, monetary_report, policy_report,
                                self.market_state, tn): item
                    for item in pool
                }
                for future in as_completed(fmap):
                    item = fmap[future]
                    try:
                        fr = future.result()
                        final_reports.append(fr)
                        print(f"  ✅ T{tn} {item['code']} {item['name']} | {fr.final_rating} | 共识:{fr.consensus_level}")
                    except Exception as e:
                        print(f"  ❌ T{tn} {item['code']} {item['name']}：{str(e)[:100]}")

        # Phase 2.5: 组合约束求解——总仓位不超过100%
        total_pos = sum(fr.suggested_position_pct for fr in final_reports)
        if total_pos > 1.0:
            scale = 1.0 / total_pos
            for fr in final_reports:
                fr.suggested_position_pct = round(fr.suggested_position_pct * scale, 4)
            print(f"\n  🔄 组合约束: 原始总仓位{total_pos*100:.0f}% → 归一化至100%")
        
        # Phase 2.6: ETF相关性约束
        self._apply_correlation_constraint(final_reports)
        
        # Phase 3: 报告输出
        ResearchReportGenerator.generate_full_report(final_reports)
        
        # 复盘：保存快照+T+1验证
        ReviewManager.process(final_reports)
        
        print(f"\n【主控Agent】全部标的处理完毕！共 {len(final_reports)} 只ETF")

    def _research_single_etf(self, item: dict, global_max_pos: float,
                               macro_report: AgentReport,
                               monetary_report: AgentReport,
                               policy_report: AgentReport,
                               market_state: str = "震荡偏强",
                               tier: int = 1) -> FinalResearchReport:
        code, name, typ, idx = item["code"], item["name"], item["type"], item["index_code"]

        if tier == 3:
            return self._rule_based_research(code, name, typ, idx, global_max_pos, macro_report, market_state)

        # Tier 1 & 2: LLM Agent analysis
        if tier == 1:
            fns = [self.value_agent.run, self.tech_agent.run, self.sentiment_agent.run,
                   self.fundflow_agent.run, self.risk_agent.run, self.industry_agent.run,
                   self.retail_sentiment_agent.run, self.cross_market_agent.run]
            fargs = [(code, name, idx), (code, name), (code, name), (code, name, idx),
                     (code, name), (code, name), (code,), (name, code)]
        else:
            fns = [self.value_agent.run, self.tech_agent.run, self.sentiment_agent.run, self.fundflow_agent.run]
            fargs = [(code, name, idx), (code, name), (code, name), (code, name, idx)]

        with ThreadPoolExecutor(max_workers=len(fns)) as agent_exec:
            futures = [agent_exec.submit(fn, *fa) for fn, fa in zip(fns, fargs)]
            reports = [f.result() for f in futures]

        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
            r.data_summary["monetary_context"] = monetary_report.analysis[:150]
            r.data_summary["policy_context"] = policy_report.analysis[:150]

        debates = []
        if tier == 1:
            disagreements = DebateEngine.detect_disagreements(reports)
            if disagreements:
                debates = DebateEngine.hold_debate(disagreements, reports, name, code)

        etf_info = {"code": code, "name": name, "type": typ, "index_code": idx}
        final_report = self.chief_agent.run(reports, debates, global_max_pos, etf_info, market_state)
        final_report.macro_context = macro_report.analysis[:200]
        return final_report

# ====================== 程序入口 ======================
if __name__ == "__main__":
    print(f"LLM多智能体模式: {'开启' if LLM_ENABLED else '关闭（规则评分模式）'}")
    print(f"辩论功能: {'开启' if DEBATE_ENABLED else '关闭'}")
    if LLM_ENABLED:
        print(f"LLM模型: {LLM_MODEL} | API: {LLM_BASE_URL}")
        print(f"多智能体辩论: {'开启(LLM驱动)' if DEBATE_ENABLED else '关闭'}")
        print(f"评分模式: z-score加权 + 凯利公式仓位 + 动态准确率权重")
    else:
        print("提示：设置 LLM_ENABLED=True 并配置有效API密钥以启用LLM深度分析")
    print()
    
    scheduler = MainSchedulerAgent()
    scheduler.run()