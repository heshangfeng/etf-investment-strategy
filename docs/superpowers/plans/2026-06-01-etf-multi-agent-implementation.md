# ETF多智能体投研系统 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `etf-agent.py` 从管道式评分系统改造为LLM多角色专家辩论式ETF多智能体投研系统

**Architecture:** 三段式架构——Phase1 独立研究（8个LLM Agent并行）→ Phase2 矛盾检测&选择性辩论 → Phase3 首席综合决策。LLM失败自动fallback到现有规则评分。

**Tech Stack:** Python, akshare, OpenAI SDK, ThreadPoolExecutor, dataclasses

**Modifies only:** `etf-agent.py`（单文件改造）

---

### Task 1: 添加 DataClass 定义

**Files:**
- Modify: `etf-agent.py`（在配置区之后、数据采集层之前插入）

- [ ] **Step 1: 在配置区之后添加 AgentReport 和 FinalResearchReport**

在 `HEADERS` 配置之后、`CACHE_*` 全局变量之前插入以下代码：

```python
# ====================== 【数据类定义】 ======================
from dataclasses import dataclass, field
from typing import Optional


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
    core_logic: str = ""
    risk_summary: str = ""
    consensus_level: str = "未知"
```

- [ ] **Step 2: 验证语法**

Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"` （确保dataClasses能正确解析）

---

### Task 2: 扩展配置区 & 添加 LLM 总开关

**Files:**
- Modify: `etf-agent.py`（配置区）

- [ ] **Step 1: 在 LLM 配置区添加新配置项**

将原有LLM配置：

```python
# 4. 大模型配置（替换为你的API信息）
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "https://api.example.com/v1"
LLM_MODEL = "gpt-3.5-turbo"
```

替换为：

```python
# 4. 大模型配置（替换为你的API信息）
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "https://api.example.com/v1"
LLM_MODEL = "gpt-4o-mini"           # 推荐性价比模型
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.3
LLM_ENABLED = True                   # 总开关：False=纯规则评分模式

# 6. 多智能体辩论配置
DEBATE_ENABLED = True
DEBATE_ROUNDS = 2
DISAGREEMENT_SCORE_THRESHOLD = 25    # 评分差异≥此值触发辩论
DISAGREEMENT_RATING_GAP = 2          # 评级级差≥此值触发辩论

RATING_ORDER = ["强烈看空", "看空", "中性", "看多", "强烈看多"]
```

- [ ] **Step 2: 验证语法**

Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 3: 添加 BaseLLMAgent 基类

**Files:**
- Modify: `etf-agent.py`（在 `PublicOpinionAgent` 类之后、`ValueScoreAgent` 之前插入）

- [ ] **Step 1: 在 PublicOpinionAgent 和 ValueScoreAgent 之间插入 BaseLLMAgent**

```python
# ====================== 【LLM多智能体基类】 ======================
class BaseLLMAgent:
    """所有LLM驱动Agent的基类。LLM失败时自动fallback到规则评分。"""
    
    ROLE_NAME = "基础分析师"
    SYSTEM_PROMPT = "你是一个专业的金融分析师。请基于提供的数据进行分析。"
    
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
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
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
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 4: 添加宏观分析智能体 MacroAnalystAgent

**Files:**
- Modify: `etf-agent.py`（在 BaseLLMAgent 之后）

- [ ] **Step 1: 在 BaseLLMAgent 后插入 MacroAnalystAgent**

```python
# ====================== 【LLM多智能体 - 宏观分析】 ======================
class MacroAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "宏观分析智能体"
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
                                      fallback_score=50, fallback_rating="中性",
                                      )
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 5: 添加6个标的级 LLM Agent

**Files:**
- Modify: `etf-agent.py`（在 MacroAnalystAgent 之后插入）

- [ ] **Step 1: 批量插入6个Agent类**

```python
# ====================== 【LLM多智能体 - 价值估值】 ======================
class ValueAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "价值估值智能体"
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
        if llm_out:
            return self._parse_to_report(etf_code, etf_name, llm_out, score)
        
        # Fallback: 百分位越低分越高
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, None, score, rating)


# ====================== 【LLM多智能体 - 技术趋势】 ======================
class TechAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "技术趋势智能体"
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
                               "数据不足20个交易日", ["数据不足"], confidence=0.3, source="rule_fallback")
        
        close, m5, m20 = df["close"].iloc[-1], df["ma5"].iloc[-1], df["ma20"].iloc[-1]
        vol = df["volatility"].tail(5).mean()
        
        # 规则评分
        ts = 50
        if close > m5 and close > m20: ts += 18
        if m5 > m20: ts += 12
        if close < m5 and close < m20: ts -= 25
        score = float(np.clip(ts, 0, 100))
        
        recent_5d_chg = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        data_text = (f"最新价: {close:.3f} | MA5: {m5:.3f} | MA20: {m20:.3f}\n"
                     f"均线关系: {'多头' if m5 > m20 else '空头'}排列\n"
                     f"近5日涨跌: {recent_5d_chg:.2f}%\n"
                     f"近5日平均波动率: {vol:.4f}")
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 35 else "看空" if score >= 20 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 舆情情绪】 ======================
class SentimentAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "舆情情绪智能体"
    SYSTEM_PROMPT = """你是行为金融学专家，擅长识别市场情绪。
分析框架：
1. 情绪分数>70为乐观（可能过度乐观），<30为恐慌（可能过度悲观）
2. 舆情趋势比单日分数更重要——持续回暖或持续走弱是强烈信号
3. 结合行业关键词判断是否存在实质性利好/利空
4. 关注多空交织情况——矛盾信号意味着市场分歧大
重点关注情绪极端值和趋势变化。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        # 复用现有舆情数据
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
        
        # 规则评分 (同PublicOpinionAgent逻辑)
        if score >= 70:
            fallback_rating = "强烈看多"
        elif score >= 60:
            fallback_rating = "看多"
        elif score >= 40:
            fallback_rating = "中性"
        elif score >= 30:
            fallback_rating = "看空"
        else:
            fallback_rating = "强烈看空"
        
        report = self._parse_to_report(etf_code, etf_name, llm_out, score, fallback_rating)
        report.data_summary = {
            "opinion_score": op['opinion_score'],
            "opinion_trend": op['opinion_trend'],
            "keywords": op['keywords'],
            "warn_alert": op['warn_alert']
        }
        return report


# ====================== 【LLM多智能体 - 资金流向】 ======================
class FundFlowAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "资金流向智能体"
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
        
        data_text = (f"成交量比(近10日均值): {vol_ratio:.2f}\n"
                     f"北向近5日净流入: {north/100000000:.2f}亿\n"
                     f"折溢价率: {premium*100:.3f}%\n"
                     f"资金分项(s1量比): {s1:.1f} | (s2北向): {s2:.1f}")
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = self._score_to_rating(score)
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【LLM多智能体 - 风险管理】 ======================
class RiskManagerAgent(BaseLLMAgent):
    ROLE_NAME = "风险管理智能体"
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
        
        data_text = (f"折溢价率: {premium*100:.3f}%\n"
                     f"日波动率: {vol:.4f}\n"
                     f"成交量: {liq:.0f}\n"
                     f"风险扣分原因: {'; '.join(risk_detail) if risk_detail else '无明显风险'}")
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        # 风控评分越高=越安全，评级含义不同
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 20 else "强烈看空"
        report = self._parse_to_report(etf_code, etf_name, llm_out, score, rating)
        return report


# ====================== 【LLM多智能体 - 行业纵析】 ======================
class IndustryAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "行业纵析智能体"
    SYSTEM_PROMPT = """你是深耕行业的资深研究员。
分析框架：
1. 识别ETF所属行业赛道（从名称判断）
2. 分析行业所处周期位置（初创/成长/成熟/衰退）
3. 匹配行业专属关键词判断当前景气度
4. 考虑政策催化方向和产业趋势
给出基于行业基本面的独立判断。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        # 匹配行业
        pos_words, neg_words = PublicOpinionAgent.get_industry_keywords(etf_name)
        matched_industries = [ind for ind in INDUSTRY_POS if ind in etf_name]
        industry = matched_industries[0] if matched_industries else "通用"
        
        data_text = (f"所属行业: {industry}\n"
                     f"行业正向词: {', '.join(pos_words)}\n"
                     f"行业负向词: {', '.join(neg_words)}\n"
                     f"该ETF名称包含的行业关键词: {industry if industry != '通用' else '无明显行业特征'}")
        
        # 规则评分：按匹配到的行业词数量估算情绪
        score = 50.0
        if industry != "通用":
            score = 60.0  # 有行业归属+基础分
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        report = self._parse_to_report(etf_code, etf_name, llm_out, score)
        return report
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 6: 新增 PublicOpinionAgent.llm_run 静态方法 & 修复 PublicOpinionAgent 回调

**Files:**
- Modify: `etf-agent.py`（在 PublicOpinionAgent 类中）

- [ ] **Step 1: 在 PublicOpinionAgent 类中添加 llm_run 方法**

在 `PublicOpinionAgent` 类中添加（复制 `run` 方法逻辑，但返回dict代替写入CACHE）：

```python
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
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 7: 添加辩论引擎

**Files:**
- Modify: `etf-agent.py`（在所有 Agent 类之后、DecisionAgent 之前插入）

- [ ] **Step 1: 在 IndustryAnalystAgent 之后、DecisionAgent 之前插入辩论引擎**

```python
# ====================== 【辩论引擎】 ======================
class DebateEngine:
    """矛盾检测 + 选择性辩论"""
    
    @staticmethod
    def detect_disagreements(reports: list[AgentReport]) -> list[dict]:
        """检测Agent间是否存在显著分歧"""
        disagreements = []
        
        # 按评分分组
        groups = {}
        for r in reports:
            groups.setdefault(r.agent_name, r)
        
        agent_names = list(groups.keys())
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a1, a2 = groups[agent_names[i]], groups[agent_names[j]]
                score_gap = abs(a1.score - a2.score)
                rating_gap = abs(RATING_ORDER.index(a1.rating) - RATING_ORDER.index(a2.rating))
                
                if score_gap >= DISAGREEMENT_SCORE_THRESHOLD or rating_gap >= DISAGREEMENT_RATING_GAP:
                    disagreements.append({
                        "agent_a": a1.agent_name,
                        "agent_b": a2.agent_name,
                        "rating_a": a1.rating,
                        "rating_b": a2.rating,
                        "score_a": a1.score,
                        "score_b": a2.score,
                        "score_gap": score_gap,
                        "focus": f"{a1.agent_name}({a1.rating}) vs {a2.agent_name}({a2.rating}) 存在分歧"
                    })
        
        return disagreements
    
    @staticmethod
    def hold_debate(disagreements: list[dict], reports: list[AgentReport],
                    etf_name: str, etf_code: str) -> list[dict]:
        """执行选择性辩论：只对分歧双方进行辩论"""
        if not DEBATE_ENABLED or not disagreements:
            return []
        
        debate_logs = []
        reports_dict = {r.agent_name: r for r in reports}
        
        for d in disagreements[:3]:  # 最多辩论3个分歧点
            agent_a = reports_dict.get(d["agent_a"])
            agent_b = reports_dict.get(d["agent_b"])
            if not agent_a or not agent_b:
                continue
            
            debate_log = {
                "topic": d["focus"],
                "agents": [d["agent_a"], d["agent_b"]],
                "rounds": []
            }
            
            # Round 1: A陈述 → B反驳
            a_view = f"我的评级是{d['rating_a']}({d['score_a']}分)，核心逻辑：{agent_a.analysis[:200]}"
            b_view = f"我的评级是{d['rating_b']}({d['score_b']}分)，核心逻辑：{agent_b.analysis[:200]}"
            
            debate_log["rounds"].append({
                "round": 1,
                "statement": f"{d['agent_a']}: {a_view}",
                "rebuttal": f"{d['agent_b']}: 我持不同观点。{b_view}"
            })
            
            if DEBATE_ROUNDS >= 2:
                # Round 2: B反问 → A回应
                debate_log["rounds"].append({
                    "round": 2,
                    "challenge": f"{d['agent_b']}反问：{d['agent_a']}的关键因子{d['agent_a']}是否充分考虑了{d['agent_b']}提及的风险？",
                    "response": f"{d['agent_a']}回应：我的判断基于{d['agent_a']}的数据维度。{d['agent_b']]提醒的风险我会标注，但不改变核心结论。"
                })
            
            debate_logs.append(debate_log)
        
        return debate_logs
```

注意：上面代码中的 `d['agent_b']]` 是写错的，实际代码应该是 `d['agent_b']` 。修正在 Step 2 中。

- [ ] **Step 2: 修正辩论引擎中的语法错误**

实际写入文件时修正 Round 2 中的 `d['agent_b']]` → `d['agent_b']`：

```python
"response": f"{d['agent_a']}回应：我的判断基于{d['agent_a']}的数据维度。{d['agent_b']}提醒的风险我会标注，但不改变核心结论。"
```

- [ ] **Step 3: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 8: 改造首席决策智能体 ChiefDecisionAgent

**Files:**
- Modify: `etf-agent.py`（重写现有的 DecisionAgent 类）

- [ ] **Step 1: 将现有 DecisionAgent 替换为 LLM-powered ChiefDecisionAgent**

```python
# ====================== 【首席决策智能体】 ======================
class ChiefDecisionAgent(BaseLLMAgent):
    ROLE_NAME = "首席决策智能体"
    SYSTEM_PROMPT = """你是投资委员会主席，需要综合各方观点做出最终判断。
你的职责：
1. 阅读所有智能体的独立分析报告
2. 审阅辩论记录（如有）
3. 综合不同维度的观点，考虑每个Agent的置信度和专业性
4. 对分歧点做出仲裁判断
5. 给出明确的最终评级、仓位建议和核心逻辑

评级标准：
- 强烈看多 (score>=80)：多项指标共振，核心机会
- 看多 (score>=65)：整体向好，有少量顾虑
- 中性 (score>=45)：多空均衡，等待信号
- 看空 (score>=30)：整体偏弱，谨慎回避
- 强烈看空 (score<30)：多项风险暴露，清仓回避

仓位映射：
- 强烈看多 → 重仓 (global_max_pos * 0.35)
- 看多 → 中仓 (global_max_pos * 0.25)
- 中性 → 轻仓 (global_max_pos * 0.15)
- 看空 → 观望 (global_max_pos * 0.05)
- 强烈看空 → 清仓 (0.0)"""
    
    def run(self, reports: list[AgentReport], debates: list[dict],
            global_max_pos: float, etf_info: dict) -> FinalResearchReport:
        
        # 计算共识度
        ratings = [r.rating for r in reports]
        unique_ratings = set(ratings)
        if len(unique_ratings) == 1:
            consensus = "高度一致"
        elif len(unique_ratings) <= 3:
            consensus = "基本一致" if not debates else "存在分歧"
        else:
            consensus = "严重分歧"
        
        # 准备LLM输入
        reports_text = "\n\n".join([
            f"【{r.agent_name}】评级:{r.rating} 评分:{r.score} 置信度:{r.confidence}\n"
            f"分析:{r.analysis[:300]}\n关键因子:{'; '.join(r.key_factors)}\n风险:{'; '.join(r.risk_warnings)}"
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
        
        # 规则综合评分（作为fallback）
        valid_reports = [r for r in reports if r.source == "llm"]
        if not valid_reports:
            valid_reports = reports  # 全部是规则评分
        
        avg_score = np.mean([r.score for r in valid_reports])
        
        # LLM决策
        llm_out = self._call_llm(self.SYSTEM_PROMPT + "\n" + self.SYSTEM_PROMPT, 
                                  self._build_user_prompt(etf_info['name'], etf_info['code'], data_text))
        
        if llm_out:
            final_rating = llm_out.get("rating", "中性")
            final_score = float(np.clip(llm_out.get("score", avg_score), 0, 100))
            core_logic = llm_out.get("analysis", "")
        else:
            final_rating = self._score_to_rating(avg_score)
            final_score = avg_score
            core_logic = f"【规则综合】基于{len(valid_reports)}个Agent的平均评分{avg_score:.1f}分"
        
        # 仓位映射
        pos_map = {"强烈看多": 0.35, "看多": 0.25, "中性": 0.15, "看空": 0.05, "强烈看空": 0.0}
        pos_pct = global_max_pos * pos_map.get(final_rating, 0.1)
        
        # 仓位建议文本
        pos_text_map = {"强烈看多": "重仓", "看多": "中仓", "中性": "轻仓", "看空": "观望", "强烈看空": "清仓回避"}
        
        # 风险汇总
        all_risks = []
        for r in reports:
            all_risks.extend(r.risk_warnings)
        all_risks = list(set(all_risks))[:5]
        
        return FinalResearchReport(
            etf_info=etf_info,
            macro_context="",
            agent_reports=reports,
            debates=debates,
            final_rating=final_rating,
            position_suggestion=pos_text_map.get(final_rating, "观望"),
            suggested_position_pct=round(pos_pct, 2),
            core_logic=core_logic,
            risk_summary="; ".join(all_risks) if all_risks else "暂无显著风险提示",
            consensus_level=consensus
        )
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 9: 改造报告输出 ResearchReportGenerator

**Files:**
- Modify: `etf-agent.py`（重写 ReportAgent 类）

- [ ] **Step 1: 替换 ReportAgent 为 ResearchReportGenerator**

```python
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
        
        # ====== 1. 摘要看板 ======
        print("\n" + "="*160)
        print("【📊 投研摘要看板】")
        print("="*160)
        
        summary_rows = []
        for fr in all_reports:
            summary_rows.append({
                "代码": fr.etf_info['code'],
                "名称": fr.etf_info['name'],
                "类型": fr.etf_info['type'],
                "首席评级": fr.final_rating,
                "建议仓位": fr.position_suggestion,
                "共识度": fr.consensus_level,
                "仓位比例": f"{fr.suggested_position_pct*100:.0f}%"
            })
        
        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))
        
        # ====== 2. 推荐分层 ======
        core = [fr for fr in all_reports if fr.final_rating in ("强烈看多", "看多")]
        watch = [fr for fr in all_reports if fr.final_rating == "中性"]
        avoid = [fr for fr in all_reports if fr.final_rating in ("看空", "强烈看空")]
        
        if core:
            print(f"\n🔥 【重点关注（{len(core)}只）】")
            for fr in core:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.final_rating} | {fr.core_logic[:100]}")
        
        if watch:
            print(f"\n👀 【跟踪观察（{len(watch)}只）】")
            for fr in watch:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 共识:{fr.consensus_level} | {fr.core_logic[:80]}")
        
        if avoid:
            print(f"\n❌ 【风险回避（{len(avoid)}只）】")
            for fr in avoid:
                print(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        
        # ====== 3. 详细报告 ======
        print("\n" + "="*160)
        print("【📋 各标的多智能体详细报告】")
        print("="*160)
        
        for fr in all_reports:
            ResearchReportGenerator._print_detailed_report(fr)
        
        # ====== 4. 板块平均 ======
        self = ResearchReportGenerator  # alias for static context
        print("\n" + "="*160)
        print("【📈 板块综合评级】")
        print("="*160)
        sector_data = {}
        for fr in all_reports:
            st = fr.etf_info['type']
            sector_data.setdefault(st, []).append(fr.final_rating)
        for sector, ratings_list in sorted(sector_data.items()):
            scores = []
            for rt in ratings_list:
                scores.append(RATING_ORDER.index(rt) if rt in RATING_ORDER else 2)
            avg_idx = np.mean(scores)
            avg_rating = RATING_ORDER[int(round(avg_idx))]
            print(f"  {sector}: {avg_rating}（{len(ratings_list)}只标的）")
        
        # ====== 5. 保存台账 ======
        save_path = f"/mnt/ETF_多智能体投研报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df_summary.to_excel(save_path, index=False)
        print(f"\n✅ 投研摘要已保存：{save_path}")
    
    @staticmethod
    def _print_detailed_report(fr: FinalResearchReport):
        """打印单只ETF的详细投研报告"""
        print(f"\n{'─'*160}")
        print(f"  {fr.etf_info['code']} {fr.etf_info['name']}（{fr.etf_info['type']}）")
        print(f"  首席评级: {fr.final_rating} | 建议: {fr.position_suggestion} ({fr.suggested_position_pct*100:.0f}%) | 共识: {fr.consensus_level}")
        print(f"{'─'*160}")
        
        # Agent报告
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
        
        # 辩论记录
        if fr.debates:
            print(f"\n  ⚔️ 【辩论记录】")
            for d in fr.debates:
                print(f"    🎯 {d['topic']}")
                for rd in d['rounds']:
                    for k, v in rd.items():
                        if k != 'round':
                            print(f"    {v[:200]}")
        
        # 首席总结
        print(f"\n  📌 【首席决策】")
        print(f"  最终评级: {fr.final_rating}")
        print(f"  核心逻辑: {fr.core_logic[:200]}")
        print(f"  综合风险: {fr.risk_summary[:200]}")
```

- [ ] **Step 2: 验证语法**

注意：上面有一个 `self = ResearchReportGenerator` 的临时别名写法在静态方法中不合适。实际代码需要修正：

在 `generate_full_report` 方法中，将板块平均部分改为直接调用 `ResearchReportGenerator._print_detailed_report`，移除 `self = ...` 别名。

```python
        # ====== 3. 详细报告 ======
        ...
        for fr in all_reports:
            ResearchReportGenerator._print_detailed_report(fr)
        
        # ====== 4. 板块平均 ======
        ...
```

Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 10: 改造主调度器 MainSchedulerAgent

**Files:**
- Modify: `etf-agent.py`（重写 MainSchedulerAgent.run 方法）

- [ ] **Step 1: 重构 MainSchedulerAgent**

```python
# ====================== 【顶层主控调度 - 三段式多智能体】 ======================
class MainSchedulerAgent:
    # ETF标的池（保持原样）
    ETF_POOL = [...]  # 原有29只ETF不变
    
    def __init__(self):
        self.macro_agent = MacroAnalystAgent()
        self.value_agent = ValueAnalystAgent()
        self.tech_agent = TechAnalystAgent()
        self.sentiment_agent = SentimentAnalystAgent()
        self.fundflow_agent = FundFlowAnalystAgent()
        self.risk_agent = RiskManagerAgent()
        self.industry_agent = IndustryAnalystAgent()
        self.chief_agent = ChiefDecisionAgent()
    
    def run(self):
        print("█"*160)
        print("  ETF 多智能体投研系统 | LLM多角色专家分析 + 辩论 + 首席决策")
        print("█"*160)
        
        # ====== Phase 0: 宏观分析 ======
        print(f"\n【Phase 0】宏观环境分析...")
        macro_report = self.macro_agent.run()
        print(f"  ✅ 宏观: {macro_report.rating} ({macro_report.score}分) | {macro_report.analysis[:80]}")
        global_max_pos = macro_report.score / 100  # 宏观分数映射为全局仓位上限
        
        if not LLM_ENABLED:
            print("  ⚙️ LLM开关=OFF，使用规则评分模式")
        
        print(f"\n【全局仓位上限】{global_max_pos*100:.0f}%")
        print(f"【外层并发】{MAIN_WORKERS} | 【Agent并发】{AGENT_WORKERS}\n")
        
        # ====== Phase 1: 独立研究 ======
        print("【Phase 1】各Agent独立研究（并行）...\n")
        
        final_reports = []
        
        with ThreadPoolExecutor(max_workers=MAIN_WORKERS) as main_exec:
            future_map = {
                main_exec.submit(self._research_single_etf, item, global_max_pos, macro_report): item
                for item in self.ETF_POOL
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    fr = future.result()
                    final_reports.append(fr)
                    print(f"  ✅ {item['code']} {item['name']} | {fr.final_rating} | 共识:{fr.consensus_level}")
                except Exception as e:
                    print(f"  ❌ {item['code']} {item['name']}：{str(e)[:100]}")
        
        # ====== Phase 3: 报告输出 ======
        ResearchReportGenerator.generate_full_report(final_reports)
        print(f"\n【主控Agent】全部标的处理完毕！共 {len(final_reports)} 只ETF")
    
    def _research_single_etf(self, item: dict, global_max_pos: float, 
                               macro_report: AgentReport) -> FinalResearchReport:
        """单只ETF的完整三段式研究流程"""
        code, name, typ, idx = item["code"], item["name"], item["type"], item["index_code"]
        
        # Phase 1: 7个Agent并行独立研究
        with ThreadPoolExecutor(max_workers=AGENT_WORKERS) as agent_exec:
            f_val = agent_exec.submit(self.value_agent.run, code, name, idx)
            f_tech = agent_exec.submit(self.tech_agent.run, code, name)
            f_sent = agent_exec.submit(self.sentiment_agent.run, code, name)
            f_fund = agent_exec.submit(self.fundflow_agent.run, code, name, idx)
            f_risk = agent_exec.submit(self.risk_agent.run, code, name)
            f_ind = agent_exec.submit(self.industry_agent.run, code, name)
            
            reports = [f.result() for f in [f_val, f_tech, f_sent, f_fund, f_risk, f_ind]]
        
        # 注入宏观上下文
        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
        
        # Phase 2: 矛盾检测 + 选择性辩论
        disagreements = DebateEngine.detect_disagreements(reports)
        debates = []
        if disagreements:
            debates = DebateEngine.hold_debate(disagreements, reports, name, code)
        
        # Phase 3: 首席决策
        etf_info = {"code": code, "name": name, "type": typ, "index_code": idx}
        final = self.chief_agent.run(reports, debates, global_max_pos, etf_info)
        final.macro_context = macro_report.analysis[:200]
        
        return final
```

- [ ] **Step 2: 验证语法**
Run: `python -c "import ast; ast.parse(open('etf-agent.py').read()); print('语法OK')"`

---

### Task 11: 清理旧类 & 最终整合

**Files:**
- Modify: `etf-agent.py`

- [ ] **Step 1: 注释/移除不再被主调度器直接调用的旧类**

保留以下类不删除（作为底层工具函数被新版Agent内调用）：
- `DataCollectAgent` ✅（被各类Agent调用）
- `PublicOpinionAgent` ✅（被 SentimentAnalystAgent 和 IndustryAnalystAgent 调用）
- `LLMInterpretAgent` ✅（可保留，作为备选）

以下类可保留但在新版中不再直接使用（建议保留注释，不删除以确保向后兼容）：
- `ValueScoreAgent`（被ValueAnalystAgent内fallback调用）
- `BoomScoreAgent`（保留，可被其他组件引用）
- `TechScoreAgent`（被TechAnalystAgent内fallback调用）
- `FundScoreAgent`（被FundFlowAnalystAgent内引用）
- `RiskScoreAgent`（被RiskManagerAgent内引用）
- `BacktestAgent`（保留，可单独使用）
- `single_etf_all_agents` 函数（可删除或注释掉——新版由 `_research_single_etf` 替代）

**注意：** 规则评分类的评分逻辑已内联到各LLM Agent的fallback中。这些旧类保留但不再从调度器直接引用。

- [ ] **Step 2: 在文件末尾添加 LLM 开关说明**

```python
# ====================== 程序入口 ======================
if __name__ == "__main__":
    print(f"LLM多智能体模式: {'开启' if LLM_ENABLED else '关闭（规则评分模式）'}")
    print(f"辩论功能: {'开启' if DEBATE_ENABLED else '关闭'}")
    if LLM_ENABLED:
        print(f"LLM模型: {LLM_MODEL} | API: {LLM_BASE_URL}")
    else:
        print("提示：设置 LLM_ENABLED=True 并配置有效API密钥以启用LLM分析")
    
    scheduler = MainSchedulerAgent()
    scheduler.run()
```

- [ ] **Step 3: 完整语法检查**

Run: `python -m py_compile etf-agent.py`
Expected: Exit code 0，无输出

---

### Task 12: 安装依赖 & 功能验证

**Files:** None (仅运行)

- [ ] **Step 1: 确保依赖齐全**

Run: `python install.py`

- [ ] **Step 2: 试运行（2只ETF快速验证，LLM关闭模式）**

临时设置文件中 `LLM_ENABLED = False`，在 `MainSchedulerAgent.run()` 中将 `ETF_POOL` 临时改为前2只ETF，然后运行：

Run: `python etf-agent.py`
Expected: 程序正常完成，输出投研报告

- [ ] **Step 3: 验证关键输出**

检查输出包含以下内容：
- Phase 0 宏观分析结果
- Phase 1 Agent独立研究结果（规则评分模式下标注"规则评分"）
- Phase 3 首席决策
- 投研摘要看板
- 板块综合评级
- 报告保存路径

- [ ] **Step 4: 恢复 ETF_POOL 完整列表**

撤销调试用的 ETF_POOL 截断，恢复全部29只ETF。

---

### 实施顺序总览

```
Task 1  → DataClass定义
Task 2  → 配置扩展
Task 3  → BaseLLMAgent基类
Task 4  → MacroAnalystAgent
Task 5  → 6个标的级Agent（一次插入）
Task 6  → PublicOpinionAgent.llm_run
Task 7  → 辩论引擎
Task 8  → ChiefDecisionAgent
Task 9  → ResearchReportGenerator
Task 10 → MainSchedulerAgent
Task 11 → 清理&整合
Task 12 → 安装&验证
```

顺序依赖：必须按 Task 1-11 的顺序执行（每个Task都在前一个的基础上添加/改造代码）
