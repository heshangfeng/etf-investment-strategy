"""
ETF 智能投资分析系统 - 数据采集与规则评分智能体
"""
import akshare as ak
import pandas as pd
import numpy as np
import time
import re
import json
import os
import glob
import logging
import jieba
import requests
from bs4 import BeautifulSoup
from sentiment_skill import FinBertSentiment
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

logger = logging.getLogger(__name__)

from config import (
    REQUEST_DELAY, HEADERS,
    CACHE_ETF_PRICE, CACHE_INDEX_VAL, CACHE_ETF_PREMIUM,
    CACHE_NORTH_CAP, CACHE_MARKET_VOL, CACHE_OPINION, CACHE_OPINION_HIST,
    CACHE_IVIX,
    PREMIUM_RISK_THRESHOLD, VOL_RISK_THRESHOLD, LIQ_THRESHOLD,
    OPINION_WARN_THRESHOLD, TREND_DAY_COUNT,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    AGENT_WORKERS, WEIGHT,
    FIN_API_KEY, FIN_API_URL,
)
from cache import PersistentCache
from keywords import BASE_POS_KEYWORDS, BASE_NEG_KEYWORDS, INDUSTRY_POS, INDUSTRY_NEG

_CACHE = PersistentCache("misc", default_ttl=3600)


# ====================== 【1. 基础数据采集智能体】 ======================
class DataCollectAgent:
    @staticmethod
    def get_market_total_volume() -> float:
        cached = CACHE_MARKET_VOL.get("value", 0)
        if cached > 0 and CACHE_MARKET_VOL.is_fresh("value", 1800):
            return cached
        try:
            # SSE成交金额(亿)
            sse = ak.stock_sse_deal_daily()
            sse_vol = float(sse.loc[sse["单日情况"] == "成交金额", "股票"].values[0])
            # SZSE成交金额(元→亿)
            szse = ak.stock_szse_summary()
            szse_vol = float(szse.loc[szse["证券类别"] == "股票", "成交金额"].values[0]) / 1e8
            vol = round(sse_vol + szse_vol, 1)
        except Exception:
            if cached > 0:
                return cached
            vol = 7000.0  # fallback中性值
        CACHE_MARKET_VOL["value"] = vol
        return vol

    @staticmethod
    def _etf_code_with_prefix(code: str) -> str:
        """ETF代码转新浪格式：510300 → sh510300, 159915 → sz159915"""
        # 沪市ETF: 51xxxx, 56xxxx, 58xxxx; 深市ETF: 159xxx
        if code.startswith(("51", "56", "58")):
            return f"sh{code}"
        return f"sz{code}"

    @staticmethod
    def get_etf_price(etf_code: str) -> pd.DataFrame:
        code = DataCollectAgent._etf_code_with_prefix(etf_code)
        if CACHE_ETF_PRICE.is_fresh(etf_code, 3600):
            return CACHE_ETF_PRICE[etf_code]
        time.sleep(REQUEST_DELAY)
        try:
            df = ak.fund_etf_hist_sina(code)
            if df is None or df.empty:
                raise ValueError(f"Empty data for {code}")
            df.columns = [str(c).strip() for c in df.columns]
            df = df.sort_values("date").reset_index(drop=True)
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            df["volatility"] = df["close"].pct_change().abs()
            df = df.bfill().ffill()
            CACHE_ETF_PRICE[etf_code] = df
            return df
        except Exception as e:
            logger.warning("get_etf_price(%s) failed: %s", etf_code, e)
            if etf_code in CACHE_ETF_PRICE:
                return CACHE_ETF_PRICE[etf_code]
            raise

    @staticmethod
    def get_index_val(index_code: str) -> dict:
        time.sleep(REQUEST_DELAY)
        if CACHE_INDEX_VAL.is_fresh(index_code, 7200):
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
        return result

    @staticmethod
    def get_etf_premium(etf_code: str) -> float:
        time.sleep(REQUEST_DELAY)
        if CACHE_ETF_PREMIUM.is_fresh(etf_code, 3600):
            return CACHE_ETF_PREMIUM[etf_code]
        try:
            df = ak.fund_etf_premium()
            matched = df[df["代码"] == etf_code]
            if not matched.empty:
                premium = matched["折溢价率"].iloc[0] / 100
            else:
                logger.warning("get_etf_premium(%s): code not found in premium data", etf_code)
                premium = 0.0
        except:
            if etf_code in CACHE_ETF_PREMIUM:
                return CACHE_ETF_PREMIUM[etf_code]
            premium = 0.0
        CACHE_ETF_PREMIUM[etf_code] = premium
        return premium

    @staticmethod
    def get_north_flow(index_code: str) -> float:
        time.sleep(REQUEST_DELAY)
        if CACHE_NORTH_CAP.is_fresh(index_code, 7200):
            return CACHE_NORTH_CAP[index_code]
        try:
            df = ak.stock_hsgt_fund_flow(symbol=index_code)
            flow = df["北向净流入"].tail(5).sum()
        except Exception as e:
            logger.warning("get_north_flow(%s) API failed: %s", index_code, e)
            if index_code in CACHE_NORTH_CAP:
                return CACHE_NORTH_CAP[index_code]
            flow = 0
        CACHE_NORTH_CAP[index_code] = flow
        return flow

    @staticmethod
    def get_margin_balance() -> dict:
        """融资融券余额（两融情绪指标）"""
        if _CACHE.is_fresh("margin", 3600):
            return _CACHE["margin"]
        try:
            szse = ak.stock_margin_detail_szse()
            sse = ak.stock_margin_detail_sse()
            result = {
                "szse_margin": float(szse["融资余额"].iloc[-1] / 1e8) if "融资余额" in szse.columns else 0,
                "sse_margin": float(sse["融资余额"].iloc[-1] / 1e8) if "融资余额" in sse.columns else 0,
                "szse_short": float(szse["融券余额"].iloc[-1] / 1e8) if "融券余额" in szse.columns else 0,
                "sse_short": float(sse["融券余额"].iloc[-1] / 1e8) if "融券余额" in sse.columns else 0,
            }
        except Exception as e:
            logger.warning("get_margin_balance API failed: %s", e)
            if "margin" in _CACHE:
                return _CACHE["margin"]
            return {"szse_margin": 0, "sse_margin": 0, "szse_short": 0, "sse_short": 0}
        _CACHE["margin"] = result
        return result

    @staticmethod
    def get_bond_yield() -> dict:
        """中美国债收益率"""
        if _CACHE.is_fresh("bond", 3600):
            return _CACHE["bond"]
        try:
            df = ak.bond_zh_us_rate()
            cn10y = float(df[df["指标名称"] == "中国国债收益率10年"]["收益率"].iloc[-1])
            us10y = float(df[df["指标名称"] == "美国国债收益率10年"]["收益率"].iloc[-1])
            result = {"cn_10y": cn10y, "us_10y": us10y, "spread": cn10y - us10y}
        except Exception as e:
            logger.warning("get_bond_yield API failed: %s", e)
            if "bond" in _CACHE:
                result = _CACHE["bond"]
            else:
                result = {"cn_10y": 2.5, "us_10y": 4.0, "spread": -1.5}
        _CACHE["bond"] = result
        return result

    @staticmethod
    def get_sector_fund_flow() -> dict:
        """板块资金流向"""
        if _CACHE.is_fresh("sector_flow", 3600):
            return _CACHE["sector_flow"]
        try:
            df = ak.stock_sector_fund_flow_summary()
            sector_map = {}
            for _, row in df.iterrows():
                sector_map[row["板块名称"]] = {
                    "流入": float(row.get("主力净流入-净额", 0)),
                    "流入排名": int(row.get("主力净流入-排名", 99))
                }
            result = sector_map
        except Exception as e:
            logger.warning("get_sector_fund_flow API failed: %s", e)
            if "sector_flow" in _CACHE:
                result = _CACHE["sector_flow"]
            else:
                result = {"板块名称": {"流入": 0, "流入排名": 99}}
        _CACHE["sector_flow"] = result
        return result

    @staticmethod
    def get_ivix(etf_code: str) -> float:
        """获取ETF对应指数的隐含波动率(VIX-like)。7200s TTL，失败返回25.0。"""
        if CACHE_IVIX.is_fresh(etf_code, 7200):
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
        except Exception as e:
            logger.warning("get_ivix(%s) API failed: %s", etf_code, e)
            if etf_code in CACHE_IVIX:
                return CACHE_IVIX[etf_code]
            val = 25.0
        CACHE_IVIX[etf_code] = val
        return val

    @staticmethod
    def get_futures_basis() -> dict:
        """获取股指期货基差。7200s TTL，失败降级使用缓存。"""
        if _CACHE.is_fresh("futures_basis", 3600):
            return _CACHE["futures_basis"]
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
        except Exception as e:
            logger.warning("get_futures_basis API failed: %s", e)
            if "futures_basis" in _CACHE:
                result = _CACHE["futures_basis"]
            else:
                result = {}
        _CACHE["futures_basis"] = result
        return result


# ====================== 【2. 宏观研判智能体（规则版）】 ======================
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
        """多源财经新闻获取：NewsNow API → 专业财经API → 新浪爬虫"""
        # 1. 尝试 NewsNow 多源聚合（来自 alphaear-news skill）
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            session = requests.Session()
            retries = Retry(total=1, backoff_factor=0.5)
            session.mount("https://", HTTPAdapter(max_retries=retries))
            resp = session.get(
                f"https://newsnow.busiyi.world/{keyword}",
                headers=HEADERS, timeout=5,
            )
            if resp.status_code == 200:
                items = resp.json().get("data", [])[:3]
                if items:
                    return "".join(f"{n.get('title', '')}。" for n in items)
        except Exception:
            pass

        # 2. 专业财经API
        if FIN_API_KEY:
            try:
                params = {"key": FIN_API_KEY, "q": keyword, "limit": 3}
                resp = requests.get(FIN_API_URL, params=params, timeout=8)
                data = json.loads(resp.text)
                news = "".join([item["title"] + "。" for item in data["news"]])
                if news:
                    return news
            except Exception:
                pass

        # 3. 降级：新浪爬虫
        news_content = ""
        url = f"https://finance.sina.com.cn/search/news?q={keyword}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            news_list = soup.find_all("div", class_="result")
            for item in news_list[:3]:
                news_content += item.get_text(strip=True) + "。"
            if news_content:
                return news_content
        except Exception:
            pass

        # 4. 兜底：akshare 财新新闻
        try:
            import akshare as ak
            df = ak.stock_news_main_cx()
            if df is not None and len(df) > 0:
                matched = df[df["summary"].str.contains(keyword, na=False)]
                items = matched.head(5) if len(matched) > 0 else df.head(5)
                return "".join(f"{row['summary']}。" for _, row in items.iterrows())
        except Exception:
            pass

        return "暂无公开财经资讯"

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
        """情感打分 0~100（FinBERT 替代 SnowNLP）"""
        if not text or "暂无公开财经资讯" in text:
            return 50.0
        score, _ = FinBertSentiment.analyze(text)
        return score

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


# ====================== 【5. 旧版分项打分智能体（规则评分 fallback）】 ======================
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


class EnhancedBacktestAgent:
    """Enhanced backtest with transaction costs and more rigorous metrics.

    Backward compatible: returns all original keys plus new ones.
    Transaction cost: 0.03% ETF commission per trade (entry or exit).
    """

    TRADE_COST = 0.0003  # 0.03% ETF commission

    @staticmethod
    def run(etf_code: str, days=120) -> dict:
        snapshot_dir = "data/snapshots"
        files = sorted(glob.glob(f"{snapshot_dir}/*.json"))
        if len(files) < 2:
            return EnhancedBacktestAgent._fallback(etf_code, days)

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
            return EnhancedBacktestAgent._fallback(etf_code, days)

        date_idx = dict(zip(ds, range(len(df))))

        for sig_date in dates:
            if sig_date not in df_dates:
                return EnhancedBacktestAgent._fallback(etf_code, days)

        strategy_rets = []
        for i, sig_date in enumerate(dates):
            idx = date_idx[sig_date]
            if idx + 1 >= len(df):
                continue
            pos = 1.0 if signals[i].get("position_pct", 0) > 0 else 0.0
            underlying_ret = df.iloc[idx + 1]["close"] / df.iloc[idx]["close"] - 1
            ret = pos * (underlying_ret - EnhancedBacktestAgent.TRADE_COST)
            strategy_rets.append(ret)

        if len(strategy_rets) < 2:
            return EnhancedBacktestAgent._fallback(etf_code, days)

        arr = np.array(strategy_rets)
        n_days = len(arr)

        # Basic metrics
        win = np.sum(arr > 0)
        lose = np.sum(arr < 0)
        win_rate = win / (win + lose) if (win + lose) > 0 else 0
        profit_avg = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0
        loss_avg = abs(float(np.mean(arr[arr < 0]))) if np.any(arr < 0) else 0
        pl_ratio = profit_avg / loss_avg if loss_avg > 0 else 1
        cum = np.cumprod(1 + arr)
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))

        # Enhanced metrics
        total_return = cum[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        sharpe = float(np.mean(arr) / max(np.std(arr), 1e-10) * np.sqrt(252))
        calmar = annual_return / abs(max_dd) if max_dd != 0 else float("inf")

        # Sortino: downside deviation (only negative returns)
        downside = arr[arr < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-10
        sortino = float(np.mean(arr) / max(downside_std, 1e-10) * np.sqrt(252))

        # Trade count: number of positive position signals
        trade_count = sum(1 for sig in signals if sig.get("position_pct", 0) > 0)

        return {
            "胜率": round(win_rate * 100, 2),
            "盈亏比": round(pl_ratio, 2),
            "最大回撤": round(max_dd * 100, 2),
            "回测收益": round(total_return * 100, 2),
            "年化收益": round(annual_return * 100, 2),
            "夏普比率": round(sharpe, 2),
            "卡玛比率": round(calmar, 2),
            "索提诺比率": round(sortino, 2),
            "交易次数": trade_count,
        }

    @staticmethod
    def _fallback(etf_code: str, days=120) -> dict:
        df = DataCollectAgent.get_etf_price(etf_code).tail(days).copy()
        df["ma20"] = df["close"].rolling(20).mean()
        df["signal"] = (df["close"] > df["ma20"]).astype(int)

        # Apply transaction cost on signal changes (entry/exit)
        prev_signal = df["signal"].shift(1).fillna(0)
        entry_cost = ((prev_signal == 0) & (df["signal"] == 1)).astype(int) * EnhancedBacktestAgent.TRADE_COST
        exit_cost = ((prev_signal == 1) & (df["signal"] == 0)).astype(int) * EnhancedBacktestAgent.TRADE_COST
        df["ret"] = df["close"].pct_change()
        df["strategy_ret"] = prev_signal * df["ret"] - entry_cost - exit_cost

        arr = df["strategy_ret"].values[1:]  # skip first NaN from pct_change
        n_days = len(arr)

        win = np.sum(arr > 0)
        lose = np.sum(arr < 0)
        win_rate = win / (win + lose) if (win + lose) > 0 else 0
        profit_avg = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0
        loss_avg = abs(float(np.mean(arr[arr < 0]))) if np.any(arr < 0) else 0
        pl_ratio = profit_avg / loss_avg if loss_avg > 0 else 1
        cum = (1 + df["strategy_ret"]).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min())

        total_return = cum.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        sharpe = float(np.nanmean(arr) / max(np.nanstd(arr), 1e-10) * np.sqrt(252))
        calmar = annual_return / abs(max_dd) if max_dd != 0 else float("inf")

        downside = arr[arr < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-10
        sortino = float(np.nanmean(arr) / max(downside_std, 1e-10) * np.sqrt(252))

        # Trade count: number of entries (signal 0→1)
        trade_count = int(np.sum((prev_signal == 0) & (df["signal"] == 1)))

        return {
            "胜率": round(win_rate * 100, 2),
            "盈亏比": round(pl_ratio, 2),
            "最大回撤": round(max_dd * 100, 2),
            "回测收益": round(total_return * 100, 2),
            "年化收益": round(annual_return * 100, 2),
            "夏普比率": round(sharpe, 2),
            "卡玛比率": round(calmar, 2),
            "索提诺比率": round(sortino, 2),
            "交易次数": trade_count,
        }
