# 项目 Skills 使用说明

本项目已安装 9 个金融投研 Skill，位于 `.opencode/skills/` 目录，仅在此项目下可用。
使用 OpenCode 时，系统会根据你的提问自动匹配 Skill，也可手动触发。

---

## 🔍 alphaear-stock — 股票行情查询

**自动触发**: 查股票代码、最新价、涨跌幅、历史行情
**手动触发**: `/alphaear-stock`

支持 A 股 / 港股 / 美股，可查询：
- 股票代码匹配（"查一下 510300 的代码"）
- 实时行情（"沪深300现价多少"）
- 历史价格走势（"创业板ETF最近一个月走势"）

---

## 📰 alphaear-news — 金融新闻分析

**自动触发**: 查财经新闻、热点事件、市场趋势
**手动触发**: `/alphaear-news`

聚合多源金融新闻（微博/知乎/华尔街见闻等），支持：
- 实时热点新闻（"今天有什么金融热点"）
- 行业趋势追踪（"半导体板块最近有什么新闻"）
- 市场情绪提取（"市场情绪如何"）

---

## 💬 alphaear-sentiment — 金融情感分析

**自动触发**: 分析文本情感倾向（正面/负面/中性）
**手动触发**: `/alphaear-sentiment`

使用 FinBERT 金融预训练模型进行情感分析，支持：
- 新闻/研报情感评分（-1.0 ~ +1.0）
- 舆情情绪判断（"这篇新闻对ETF是利好还是利空"）
- 批量文本情感分析

---

## 📈 alphaear-predictor — 市场预测

**自动触发**: 预测市场走势、时间序列预测
**手动触发**: `/alphaear-predictor`

基于 Kronos 时序预测模型，支持：
- ETF 价格趋势预测（"科创50ETF下周走势"）
- 考虑新闻情绪的市场修正预测
- 多时间维度的预测（短期/中期/长期）

---

## 📊 alphaear-reporter — 研报生成

**自动触发**: 写投资分析报告、研报
**手动触发**: `/alphaear-reporter`

自动化生成专业金融分析报告，支持：
- 个股/ETF 深度研究报告
- 行业分析报告
- 图表配置自动生成
- 多格式输出

---

## 🧠 alphaear-logic-visualizer — 投资逻辑可视化

**自动触发**: 画投资逻辑图、传导链路图
**手动触发**: `/alphaear-logic-visualizer`

生成 Draw.io XML 格式的金融逻辑传导图，用于：
- 投资逻辑链路可视化（"画一下降息对ETF的传导路径"）
- Agent 打分逻辑展示
- 多因子影响路径图

---

## 📡 alphaear-signal-tracker — 信号追踪

**自动触发**: 追踪投资信号演化、更新逻辑
**手动触发**: `/alphaear-signal-tracker`

基于最新市场信息追踪投资信号变化，支持：
- 信号状态跟踪（"之前的买入信号还成立吗"）
- 信号强度变化评估
- 多信号综合分析

---

## 🔎 alphaear-search — 金融搜索 RAG

**自动触发**: 搜索金融信息、查找资料
**手动触发**: `/alphaear-search`

混合搜索（网络 + 本地知识库），用于：
- 金融知识查询（"什么是风险平价策略"）
- 研报/文档搜索
- 多源信息聚合

---

## ⚡ alphaear-deepear-lite — 信号链路分析

**自动触发**: 获取金融信号和传导链分析
**手动触发**: `/alphaear-deepear-lite`

从 DeepEar Lite 获取最新金融信号，支持：
- 信号传导链分析
- 多市场联动信号
- 实时信号更新

---

## 🛠 skill-creator — Skill 创建工具

**自动触发**: 创建或更新自定义 Skill
**手动触发**: `/skill-creator`

用于创建本项目的自定义 OpenCode Skill，支持：
- 初始化新的 SKILL.md
- 打包 Skill 脚本和依赖
- 快速验证 Skill 配置

---

## 自动触发匹配说明

当你在对话中说以下关键词时，系统会自动加载对应的 Skill：

| 关键词 | 匹配 Skill |
|--------|-----------|
| "查一下"/"代码"/"行情"/"价格" | alphaear-stock |
| "新闻"/"热点"/"消息"/"大事" | alphaear-news |
| "情感"/"情绪"/"舆情"/"看法" | alphaear-sentiment |
| "预测"/"走势"/"未来"/"涨跌" | alphaear-predictor |
| "报告"/"研报"/"分析报告" | alphaear-reporter |
| "逻辑图"/"传导"/"链路图" | alphaear-logic-visualizer |
| "信号"/"追踪"/"信号变化" | alphaear-signal-tracker |
| "搜索"/"找一下"/"查资料" | alphaear-search |
| "创建技能"/"新技能"/"skill" | skill-creator |

> 注意：这些 Skill 仅在此项目（ETF投资策略）目录下生效，切到其他项目不会加载。
