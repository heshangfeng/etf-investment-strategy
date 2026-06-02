# ETF 多智能体维度扩展 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 在现有ETF多智能体系统中新增5个分析维度Agent，提升投研覆盖度

**Architecture:** 4个新Agent（货币政策/政策事件/零售情绪/跨市场）+ 1个升级（行业轮动），均继承BaseLLMAgent，LLM失败fallback到规则评分。新增Agent注册到MainSchedulerAgent。

**Tech Stack:** Python, akshare, BaseLLMAgent（已有基类）

**Modifies only:** `etf-agent.py`

---

### Task 1: 新增 MonetaryPolicyAgent（货币政策/流动性分析）

**Files:**
- Modify: `etf-agent.py`（在最后一个LLM Agent类之后、DebateEngine之前插入）

- [ ] **Step 1: 确定插入位置**

当前代码中所有Agent类（ValueAnalystAgent ~ IndustryAnalystAgent）之后是DebateEngine。新Agent插在IndustryAnalystAgent和DebateEngine之间。

- [ ] **Step 2: 插入 MonetaryPolicyAgent 类**

锚点文本：
```
        return self._parse_to_report(etf_code, etf_name, llm_out, score)


# ====================== 【辩论引擎】 ======================
```

替换为：
```
        return self._parse_to_report(etf_code, etf_name, llm_out, score)


# ====================== 【LLM多智能体 - 货币政策/流动性】 ======================
class MonetaryPolicyAgent(BaseLLMAgent):
    ROLE_NAME = "货币政策智能体"
    SYSTEM_PROMPT = \"\"\"你是前央行研究员，专注于货币政策与流动性分析。
分析框架：
1. 政策利率（MLF、LPR、7天逆回购利率）变动趋势——降息周期 vs 加息周期
2. 存款准备金率(RRR)水平——降准释放流动性为正面信号
3. 社融/M2增速——宽信用还是紧信用
4. 银行间质押式回购利率(DR007)——短期流动性松紧
5. 人民币汇率(USDCNY)——大幅贬值制约宽松空间

判断准则：
- 降息+降准+社融回升+DR007低位 = 宽松周期，强烈利好权益ETF
- 加息+升准+社融回落+DR007高位 = 紧缩周期，利空权益ETF
- 混合信号 = 中性，需进一步观察
请基于可获得的数据给出流动性环境判断。\"\"\"
    
    def run(self) -> AgentReport:
        \"\"\"货币政策分析（全局级别，只需跑一次）\"\"\"
        try:
            # 尝试获取LPR数据
            df_lpr = ak.interest_rate_lpr()
            lpr_1y = float(df_lpr[df_lpr["期限种类"] == "1年期"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
            lpr_5y = float(df_lpr[df_lpr["期限种类"] == "5年期以上"]["利率"].iloc[0]) if len(df_lpr) > 0 else None
        except Exception:
            lpr_1y, lpr_5y = None, None
        
        try:
            # 尝试获取人民币汇率
            fx = ak.spot_quote()  # 可能不同版本不同
            usdcny = None
            if fx is not None:
                usd_row = fx[fx["名称"].str.contains("美元", na=False)]
                if len(usd_row) > 0:
                    usdcny = float(usd_row["现价"].iloc[0])
        except Exception:
            usdcny = None
        
        # 规则评分：基于可获得的数据
        score = 50.0
        liquidity_signals = []
        
        if lpr_1y is not None:
            liquidity_signals.append(f"1年期LPR: {lpr_1y}%")
            if lpr_1y <= 3.1: score += 20  # 低利率=宽松
            elif lpr_1y >= 3.85: score -= 15
        
        if lpr_5y is not None:
            liquidity_signals.append(f"5年期LPR: {lpr_5y}%")
        
        if usdcny is not None:
            liquidity_signals.append(f"USDCNY: {usdcny}")
            if usdcny > 7.3: score -= 15  # 贬值压力制约宽松
            elif usdcny < 6.8: score += 10  # 升值有利于资产价格
        
        score = float(np.clip(score, 0, 100))
        data_text = "\\n".join(liquidity_signals) if liquidity_signals else "暂无实时货币政策数据"
        
        # 构建LPR趋势描述
        trend_note = ""
        if lpr_1y is not None and lpr_5y is not None:
            if lpr_1y <= 3.1:
                trend_note = f"当前1年期LPR{lpr_1y}%，处于历史低位，货币政策宽松取向"
            else:
                trend_note = f"当前1年期LPR{lpr_1y}%，货币政策中性偏紧"
            data_text += f"\\n{trend_note}"
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国市场", "MONETARY", data_text))
        
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("MONETARY", "货币政策", llm_out, score, rating)
        report.data_summary = {"lpr_1y": lpr_1y, "lpr_5y": lpr_5y, "usdcny": usdcny}
        return report


# ====================== 【辩论引擎】 ======================
```

---

### Task 2: 新增 PolicyEventAgent（政策事件/政治周期）

**Files:**
- Modify: `etf-agent.py`（在 MonetaryPolicyAgent 之后、DebateEngine 之前）

- [ ] **Step 1: 插入 PolicyEventAgent 类**

锚点文本：
```
        return report


# ====================== 【辩论引擎】 ======================
```

注意：这个锚点是MonetaryPolicyAgent插入后的最后一个return。实际插入点是在MonetaryPolicyAgent类结束之后、DebateEngine之前。

插入：
```
# ====================== 【LLM多智能体 - 政策事件/政治周期】 ======================
class PolicyEventAgent(BaseLLMAgent):
    ROLE_NAME = "政策事件智能体"
    SYSTEM_PROMPT = \"\"\"你是资深政策分析师，专注中国政治经济周期。
分析框架：
1. 识别当前所处的政策周期阶段——重要会议前后市场通常有规律性表现
2. 重大会议窗口：两会(3月)、政治局会议(4/7/10/12月)、中央经济工作会议(12月)、三中全会
3. 行业政策催化——近期是否有针对特定行业的重大政策出台
4. 政策预期差——市场预期 vs 实际落地的差距驱动短期波动
5. 政策基调判断：宽松/中性/收紧

A股政策特征：
- 两会前后通常有"春季躁动"行情
- 政治局会议定调影响季度级别方向
- 行业政策（如集成电路大基金、新能源补贴）直接催化对应ETF
- 监管政策（如互联网反垄断、教育双减）导致行业ETF剧烈调整

基于当前时间点和可获得信息给出政策周期判断。\"\"\"
    
    # 重要会议日历（月-日）
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
        
        # 检测当前是否在重要会议窗口
        current_event = None
        days_to_event = None
        for name, sm, sd, em, ed in self.KEY_EVENTS:
            ev_start = datetime(now.year, sm, sd)
            ev_end = datetime(now.year, em, ed)
            if ev_start <= now <= ev_end:
                current_event = f"当前处于{name}窗口期"
                break
            if now < ev_start:
                days_to_event = (ev_start - now).days
                if days_to_event <= 30:
                    current_event = f"距离{name}还有{days_to_event}天"
                    break
        
        # 判断季节效应
        season_effect = ""
        if month in (1, 2, 3):
            season_effect = "春季躁动窗口"
        elif month == 4:
            season_effect = "年报季+政治局会议定调"
        elif month in (7, 8):
            season_effect = "中报季+政治局会议落地"
        elif month in (10, 11):
            season_effect = "三季报+年末政策定调"
        elif month == 12:
            season_effect = "中央经济工作会议+机构调仓"
        
        data_text = (f"当前日期: {now.strftime('%Y-%m-%d')}\\n"
                     f"季节效应: {season_effect}\\n"
                     f"会议窗口: {current_event or '无重要会议窗口'}\\n"
                     f"月份特征: {month}月")
        
        # 规则评分：会议窗口前后偏正面
        score = 55.0
        if current_event:
            score += 10
        if month in (3, 7, 12):
            score += 8  # 重要会议月
        if month in (1, 2):
            score += 5  # 春季躁动
        score = float(np.clip(score, 0, 100))
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("中国政策周期", "POLICY", data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report("POLICY", "政策周期", llm_out, score, rating)
        report.data_summary = {"current_event": current_event, "season_effect": season_effect, "month": month}
        return report


# ====================== 【辩论引擎】 ======================
```

---

### Task 3: 新增 RetailSentimentAgent（零售情绪结构）

**Files:**
- Modify: `etf-agent.py`（在 PolicyEventAgent 之后、DebateEngine 之前）

- [ ] **Step 1: 插入 RetailSentimentAgent**

插入 PolicyEventAgent 之后、DebateEngine 之前。

```
# ====================== 【LLM多智能体 - 零售情绪结构】 ======================
class RetailSentimentAgent(BaseLLMAgent):
    ROLE_NAME = "零售情绪智能体"
    SYSTEM_PROMPT = \"\"\"你是行为金融学量化分析师，专注A股散户情绪测量。
分析框架：
1. 融资融券余额（两融余额）——散户杠杆水平，高位=情绪过热，低位=恐慌
2. 市场换手率——极高换手率=情绪亢奋（危险信号），极低=冷清（可能见底）
3. 新增投资者开户数——开户激增=情绪高潮，开户低迷=底部区域
4. 成交量变化率——急速放量=情绪爆发，持续缩量=情绪冰点

散户情绪判断准则：
- 两融余额创新高+换手率>5%+开户数激增 = 情绪过热，警惕见顶
- 两融余额创新低+换手率<1%+开户数低迷 = 情绪冰点，可能见底
- 各项指标温和 = 正常市场，关注趋势变化

A股特征：散户情绪是典型的反向指标——极度乐观时见顶，极度悲观时见底。\"\"\"
    
    def run(self, etf_code: str) -> AgentReport:
        \"\"\"基于市场整体数据的散户情绪分析\"\"\"
        df = DataCollectAgent.get_etf_price(etf_code)
        
        # 换手率代理：成交量比（近20日均值）
        vol_20_mean = df["volume"].tail(20).mean()
        vol_today = df["volume"].iloc[-1]
        turnover_ratio = vol_today / vol_20_mean if vol_20_mean > 0 else 1.0
        
        # 价格位置（用来辅助判断情绪）
        close = df["close"].iloc[-1]
        high_52w = df["close"].tail(250).max() if len(df) >= 250 else df["close"].max()
        low_52w = df["close"].tail(250).min() if len(df) >= 250 else df["close"].min()
        pos_from_low = (close - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        
        # 规则评分
        score = 50.0
        sentiment_signals = []
        
        # 量比情绪
        if turnover_ratio > 2.0:
            sentiment_signals.append(f"放量异常(量比{turnover_ratio:.2f})")
            score -= 10  # 放量过大=情绪过热
        elif turnover_ratio > 1.5:
            sentiment_signals.append(f"放量(量比{turnover_ratio:.2f})")
            score += 5
        elif turnover_ratio < 0.5:
            sentiment_signals.append(f"缩量(量比{turnover_ratio:.2f})")
            score -= 5
        else:
            sentiment_signals.append(f"量能正常(量比{turnover_ratio:.2f})")
        
        # 价格位置情绪
        if pos_from_low > 90:
            sentiment_signals.append("接近年内高点")
            score -= 8  # 高位警惕
        elif pos_from_low < 10:
            sentiment_signals.append("接近年内低点")
            score += 8  # 低位可能反转
        elif 40 <= pos_from_low <= 60:
            sentiment_signals.append("价格中位区间")
            score += 3
        
        score = float(np.clip(score, 0, 100))
        data_text = (f"量比(20日均值): {turnover_ratio:.2f}\\n"
                     f"52周价格位置: {pos_from_low:.1f}%\\n"
                     f"情绪信号: {'; '.join(sentiment_signals)}")
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt("市场情绪", etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        report = self._parse_to_report(etf_code, "市场情绪", llm_out, score, rating)
        report.data_summary = {"turnover_ratio": turnover_ratio, "pos_from_low": pos_from_low}
        return report


# ====================== 【辩论引擎】 ======================
```

---

### Task 4: 新增 CrossMarketAgent（跨资产/跨市场联动）

**Files:**
- Modify: `etf-agent.py`（在 RetailSentimentAgent 之后、DebateEngine 之前）

- [ ] **Step 1: 插入 CrossMarketAgent**

```
# ====================== 【LLM多智能体 - 跨资产/跨市场联动】 ======================
class CrossMarketAgent(BaseLLMAgent):
    ROLE_NAME = "跨市场联动智能体"
    SYSTEM_PROMPT = \"\"\"你是全球宏观策略分析师，专注跨市场信号传导。
分析框架：
1. 美债收益率(10Y UST)——全球资产定价锚，上行压制成长股ETF
2. 美元指数(DXY)——美元强弱影响外资流向新兴市场
3. 人民币汇率(USDCNY)——人民币升值利好A股，贬值承压
4. 黄金价格——避险情绪指标，与风险资产负相关
5. 恒生指数/中概股——A股"先行指标"，港股情绪传导

传导逻辑：
- 美债收益上升+美元走强+人民币贬值 = 新兴市场承压，利空A股ETF
- 美债收益下降+美元走弱+人民币升值 = 新兴市场受益，利好A股ETF
- 黄金上涨+股市下跌 = 避险模式
- 恒生大涨通常领先A股1-2天反弹

请在数据有限的情况下，基于可获得信息做出合理判断。\"\"\"
    
    def run(self, etf_name: str, etf_code: str) -> AgentReport:
        signals = []
        
        # 获取汇率
        try:
            # 可以用ak.spot_quote 获取美元/人民币
            fx = ak.spot_quote()
            cny_row = fx[fx["名称"].str.contains("美元", na=False)]
            if len(cny_row) > 0:
                usdcny = float(cny_row["现价"].iloc[0])
                signals.append(f"USDCNY: {usdcny}")
        except Exception:
            usdcny = None
        
        # 规则评分
        score = 50.0
        if usdcny is not None:
            if usdcny > 7.3:
                score -= 15
                signals.append("→ 人民币贬值压力大，利空")
            elif usdcny < 6.9:
                score += 10
                signals.append("→ 人民币偏强，利好")
            else:
                signals.append("→ 汇率中性区间")
        
        score = float(np.clip(score, 0, 100))
        data_text = "\\n".join(signals) if signals else "暂无实时跨市场数据"
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 60 else "中性" if score >= 40 else "看空" if score >= 25 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)


# ====================== 【辩论引擎】 ======================
```

---

### Task 5: 升级 IndustryAnalystAgent → 加入行业轮动/相对强度

**Files:**
- Modify: `etf-agent.py`（重写 IndustryAnalystAgent 的 run 方法）

- [ ] **Step 1: 替换 IndustryAnalystAgent 的 run 方法**

找到现有 `IndustryAnalystAgent` 类，将其 `run` 方法替换为增强版（新增行业动量排名+相对强度逻辑）：

```python
class IndustryAnalystAgent(BaseLLMAgent):
    ROLE_NAME = "行业纵析智能体"
    SYSTEM_PROMPT = """你是深耕行业的资深研究员，精通行业轮动分析。
分析框架：
1. 识别ETF所属行业赛道（从名称判断）
2. 分析行业所处周期位置（初创/成长/成熟/衰退）
3. 匹配行业专属关键词判断当前景气度
4. 考虑政策催化方向和产业趋势
5. 相对强度——该ETF近期相对大盘的表现强弱
6. 行业动量——该行业在所有行业中的排名位置

A股行业轮动特征：
- 强势行业持续期通常3-6个月
- 资金会从高位行业流向低位行业（均值回归）
- 政策催化可以提前启动行业轮动
给出基于行业基本面和轮动位置的独立判断。"""
    
    def run(self, etf_code: str, etf_name: str) -> AgentReport:
        pos_words, neg_words = PublicOpinionAgent.get_industry_keywords(etf_name)
        matched_industries = [ind for ind in INDUSTRY_POS if ind in etf_name]
        industry = matched_industries[0] if matched_industries else "通用"
        
        # 获取ETF自身表现数据
        df = DataCollectAgent.get_etf_price(etf_code)
        close = df["close"].iloc[-1]
        
        # 相对强度计算：近20日涨幅
        ret_20d = (close / df["close"].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        ret_5d = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        
        # 用沪深300做基准比较（近似相对强度）
        try:
            benchmark = DataCollectAgent.get_etf_price("510300")
            bench_close = benchmark["close"].iloc[-1]
            bench_ret_20d = (bench_close / benchmark["close"].iloc[-20] - 1) * 100 if len(benchmark) >= 20 else 0
            relative_strength = ret_20d - bench_ret_20d
            strength_desc = f"跑赢大盘{relative_strength:.1f}%" if relative_strength > 0 else f"跑输大盘{abs(relative_strength):.1f}%"
        except Exception:
            relative_strength = 0
            strength_desc = "无法计算相对强度"
        
        # 行业动量：近20日涨幅
        momentum_score = 50 + ret_20d * 1.5 if ret_20d > 0 else 50 + ret_20d * 1.5
        
        data_text = (f"所属行业: {industry}\\n"
                     f"行业正向词: {', '.join(pos_words)}\\n"
                     f"行业负向词: {', '.join(neg_words)}\\n"
                     f"近5日涨幅: {ret_5d:.2f}%\\n"
                     f"近20日涨幅: {ret_20d:.2f}%\\n"
                     f"相对强度(20日vs沪深300): {strength_desc}\\n"
                     f"行业动量评分: {momentum_score:.1f}")
        
        # 规则评分：结合行业匹配度和动量
        score = 50.0
        if industry != "通用":
            score += 10  # 有明确行业归属
        score += np.clip(ret_20d * 0.5, -20, 20)  # 动量贡献
        score = float(np.clip(score, 0, 100))
        
        llm_out = self._call_llm(self.SYSTEM_PROMPT, self._build_user_prompt(etf_name, etf_code, data_text))
        rating = "强烈看多" if score >= 80 else "看多" if score >= 65 else "中性" if score >= 45 else "看空" if score >= 30 else "强烈看空"
        return self._parse_to_report(etf_code, etf_name, llm_out, score, rating)
```

---

### Task 6: 注册新Agent到 MainSchedulerAgent

**Files:**
- Modify: `etf-agent.py`（在 MainSchedulerAgent 的 `__init__` 和 `_research_single_etf` 中注册）

- [ ] **Step 1: 在 `__init__` 中实例化新Agent**

找到 `__init__` 方法，在 `self.industry_agent` 之后添加：
```python
        self.monetary_agent = MonetaryPolicyAgent()
        self.policy_agent = PolicyEventAgent()
        self.retail_sentiment_agent = RetailSentimentAgent()
        self.cross_market_agent = CrossMarketAgent()
```

- [ ] **Step 2: 在 `run()` 中运行全局Agent（货币政策和政策事件）**

在 Phase 0 宏观分析之后、Phase 1 之前添加：
```python
        # Phase 0.5: 全局政策&流动性分析
        print(f"\n【Phase 0.5】政策与流动性分析...")
        monetary_report = self.monetary_agent.run()
        print(f"  ✅ 货币政策: {monetary_report.rating} ({monetary_report.score}分)")
        policy_report = self.policy_agent.run()
        print(f"  ✅ 政策周期: {policy_report.rating} ({policy_report.score}分)")
```

- [ ] **Step 3: 在 `_research_single_etf` 中并行运行新增标的级Agent**

在 Phase 1 的 agent_exec.submit 块中添加：
```python
            f_retail = agent_exec.submit(self.retail_sentiment_agent.run, code)
            f_cross = agent_exec.submit(self.cross_market_agent.run, name, code)
```

并在结果收集行添加：
```python
            reports = [f.result() for f in [f_val, f_tech, f_sent, f_fund, f_risk, f_ind, f_retail, f_cross]]
```

- [ ] **Step 4: 将政策/货币报告注入每个ETF的research上下文**

在 `_research_single_etf` 中，在执行完reports后添加：
```python
        # 注入全局分析结果到每个Agent的数据摘要
        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
```

并在注入代码之后、Phase 2之前添加货币政策/政策事件到report的数据上下文（通过FinalResearchReport的macro_context字段已包含宏观信息，可再扩展）：

在 Phase 3 首席决策之前，修改 `etf_info` 构造或单独注入：
```python
        # 注入全局政策/货币上下文
        for r in reports:
            r.data_summary["monetary_context"] = monetary_report.analysis[:150] if 'monetary_report' in dir() else ""
            r.data_summary["policy_context"] = policy_report.analysis[:150] if 'policy_report' in dir() else ""
```

但上述 dir() 方式不优雅。更好的方式：在 `_research_single_etf` 的参数中传递 monetary_report 和 policy_report，然后在函数内使用。

所以修改 `_research_single_etf` 签名：
```python
    def _research_single_etf(self, item: dict, global_max_pos: float,
                               macro_report: AgentReport,
                               monetary_report: AgentReport,
                               policy_report: AgentReport) -> FinalResearchReport:
```

并在调用处 `main_exec.submit(self._research_single_etf, item, global_max_pos, macro_report, monetary_report, policy_report)`。

在函数体内注入：
```python
        for r in reports:
            r.data_summary["macro_context"] = macro_report.analysis[:200]
            r.data_summary["monetary_context"] = monetary_report.analysis[:150]
            r.data_summary["policy_context"] = policy_report.analysis[:150]
```

---

### Task 7: 语法验证

**Files:** None

- [ ] **Step 1: 运行语法检查**

Run: `python -X utf8 -c "import ast; ast.parse(open('etf-agent.py', encoding='utf-8').read()); print('Syntax OK')"`

- [ ] **Step 2: 功能验证（2只ETF快速跑）**

临时设置 ETF_POOL 为前2只ETF，运行：
Run: `python -X utf8 etf-agent.py`
Expected: 宏观分析完成，Phase 0.5 显示货币政策+政策周期结果，Phase 1 显示8个Agent结果（含零售情绪+跨市场），Phase 3 显示完整报告。

- [ ] **Step 3: 恢复 ETF_POOL 为完整29只**

--- 

### 实施顺序

```
Task 1 → MonetaryPolicyAgent（插入）
Task 2 → PolicyEventAgent（插入）
Task 3 → RetailSentimentAgent（插入）
Task 4 → CrossMarketAgent（插入）
Task 5 → 升级 IndustryAnalystAgent（替换run方法）
Task 6 → 注册到 MainSchedulerAgent
Task 7 → 验证
```

注意：Task 1-4 的插入位置都是 IndustryAnalystAgent 和 DebateEngine 之间。它们依次插入，所以插入锚点会变化。建议逐个执行，每个Task确认插入正确后再继续下一个。
