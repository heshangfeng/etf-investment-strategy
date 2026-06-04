"""
ETF 智能投资分析系统 - 工作流总监 Agent

每次运行后自动审查各 Agent 输出质量，发现盲点，累积优化建议。
"""
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from core.config import OUTPUT_DIR, REVIEW_DIR

from infra.logger import get_logger; logger = get_logger(__name__)

OPTIMIZATION_FILE = Path(REVIEW_DIR) / "optimization.json"
REVIEW_STATS_FILE = Path(REVIEW_DIR) / "cumulative_stats.json"


class WorkflowDirector:
    """
    工作流总监：审查 Agent 输出 → 发现盲点 → 累积优化建议 → 反馈到下次运行。
    
    审查维度：
    1. Agent 覆盖完整性：每只 ETF 是否凑齐 12 个 Agent 报告
    2. 分析质量：分析文本是否实质性（>50字）
    3. 数据新鲜度：分析中是否有"暂无、无数据"等信号
    4. 数据管道健康：Rule Fallback占比、配置参数合理性（TREND_DAY_COUNT等）
    5. 历史准确率趋势：读取 ReviewManager 累积数据，跟踪准确率变化
    6. 评分校准：高分→买入、低分→卖出的映射是否合理
    7. Agent 冗余：是否有 Agent 对持续产出相近评分
    8. 辩论价值：辩论是否实质性影响了最终决策
    9. LLM 覆盖规则率：系统是否过度依赖 LLM 判断
    10. 仓位合理性：单只 >25% 集中度风险、总仓位 >100% 异常
    """

    @classmethod
    def review(cls, all_reports: list) -> dict:
        """
        对一次完整的分析运行进行质量审查。
        
        Args:
            all_reports: FinalResearchReport 列表
        
        Returns:
            本次审查发现 dict
        """
        findings = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_etfs": len(all_reports),
            "issues": [],
            "suggestions": [],
            "data_quality": {},
        }

        # ── 1. 检查每只 ETF 的 Agent 覆盖情况 ──
        for fr in all_reports:
            code = fr.etf_info.get("code", "")
            name = fr.etf_info.get("name", "")
            agent_count = len(fr.agent_reports)
            
            # 检查 Agent 数量是否达标（应有 12 个）
            if agent_count < 12:
                findings["issues"].append({
                    "type": "agent_missing",
                    "severity": "high",
                    "etf": f"{name}({code})",
                    "detail": f"仅有 {agent_count}/12 个 Agent 报告",
                })

            # 检查是否有过于空泛的分析
            for ar in fr.agent_reports:
                analysis_len = len(ar.analysis)
                if analysis_len < 50:
                    findings["issues"].append({
                        "type": "generic_analysis",
                        "severity": "medium",
                        "agent": ar.agent_name,
                        "etf": f"{name}({code})",
                        "detail": f"分析仅 {analysis_len} 字，内容空泛",
                    })

                # 检查置信度是否合理
                if ar.confidence < 0.2:
                    findings["issues"].append({
                        "type": "low_confidence",
                        "severity": "low",
                        "agent": ar.agent_name,
                        "etf": f"{name}({code})",
                        "detail": f"置信度仅 {ar.confidence:.2f}",
                    })

                # 检查关键因子是否为空
                if not ar.key_factors:
                    findings["issues"].append({
                        "type": "no_key_factors",
                        "severity": "medium",
                        "agent": ar.agent_name,
                        "etf": f"{name}({code})",
                        "detail": "关键因子为空",
                    })

        # ── 2. 检查整体评级分布是否异常 ──
        ratings = [fr.final_rating for fr in all_reports]
        rating_counts = {}
        for r in ratings:
            rating_counts[r] = rating_counts.get(r, 0) + 1
        
        total = len(ratings)
        if total > 0:
            # 如果超过 80% 集中在同一评级，可能是系统性偏斜
            for rating, count in rating_counts.items():
                pct = count / total * 100
                if pct > 80:
                    findings["issues"].append({
                        "type": "rating_bias",
                        "severity": "high",
                        "detail": f"{rating} 占比 {pct:.0f}%，可能存在系统性偏斜",
                    })

        # ── 3. 数据新鲜度检查（通过 data_text 分析） ──
        for fr in all_reports:
            for ar in fr.agent_reports:
                # 检查分析文本中是否包含"暂无"、"无数据"、"无法获取"等关键词
                no_data_patterns = ["暂无", "无数据", "无法获取", "数据不足", "缺失"]
                for pattern in no_data_patterns:
                    if pattern in ar.analysis:
                        findings["issues"].append({
                            "type": "data_missing",
                            "severity": "medium",
                            "agent": ar.agent_name,
                            "etf": f"{fr.etf_info.get('name','')}({fr.etf_info.get('code','')})",
                            "detail": f"包含'{pattern}'，可能数据源不可用",
                        })
                        break

        # ── 4. 数据管道健康检查（新增） ──
        cls._check_data_pipeline(all_reports, findings)

        # ── 5. 历史准确率分析（基于 ReviewManager 累积数据） ──
        cls._check_historical_accuracy(findings)

        # ── 6. 评分校准检查 ──
        cls._check_calibration(all_reports, findings)

        # ── 7. Agent 冗余检测 ──
        cls._check_agent_redundancy(all_reports, findings)

        # ── 8. 辩论影响分析 ──
        cls._check_debate_impact(all_reports, findings)

        # ── 9. LLM 覆盖规则率跟踪 ──
        cls._check_llm_override_rate(all_reports, findings)

        # ── 10. 仓位合理性检查 ──
        cls._check_position_sizing(all_reports, findings)

        # ── 11. 生成优化建议 ──
        cls._generate_suggestions(findings)

        # ── 12. 保存结果（累积） ──
        cls._save(findings)

        return findings

    @classmethod
    def _generate_suggestions(cls, findings: dict):
        suggestions = []
        suggestions.extend(cls._suggest_agent_missing(findings))
        suggestions.extend(cls._suggest_generic_analysis(findings))
        suggestions.extend(cls._suggest_data_missing(findings))
        suggestions.extend(cls._suggest_rating_bias(findings))
        suggestions.extend(cls._suggest_agent_underperforming(findings))
        suggestions.extend(cls._suggest_calibration(findings))
        suggestions.extend(cls._suggest_agent_redundancy(findings))
        suggestions.extend(cls._suggest_llm_override(findings))
        suggestions.extend(cls._suggest_position_concentration(findings))
        suggestions.extend(cls._suggest_total_position(findings))
        suggestions.extend(cls._suggest_debate_low_impact(findings))
        suggestions.extend(cls._suggest_accuracy_declining(findings))
        suggestions.extend(cls._suggest_data_pipeline(findings))
        findings["suggestions"] = suggestions

    @classmethod
    def _suggest_agent_missing(cls, findings: dict) -> list[dict]:
        agent_missing = [i for i in findings.get("issues", []) if i["type"] == "agent_missing"]
        if not agent_missing:
            return []
        return [{
            "target": "scheduler.py",
            "action": "检查 Agent 并行执行是否有超时或异常退出",
            "priority": "high",
            "reason": f"发现 {len(agent_missing)} 只 ETF 的 Agent 数量不足",
        }]

    @classmethod
    def _suggest_generic_analysis(cls, findings: dict) -> list[dict]:
        generic = [i for i in findings.get("issues", []) if i["type"] == "generic_analysis"]
        if not generic:
            return []
        by_agent = {}
        for i in generic:
            a = i["agent"]
            by_agent.setdefault(a, []).append(i)
        return [{
            "target": f"agents 中的 {agent}",
            "action": f"增强 {agent} 的 SYSTEM_PROMPT，要求输出更详细的分析（至少 100 字）",
            "priority": "medium",
            "reason": f"发现 {len(issues)} 次空泛分析",
        } for agent, issues in by_agent.items()]

    @classmethod
    def _suggest_data_missing(cls, findings: dict) -> list[dict]:
        data_missing = [i for i in findings.get("issues", []) if i["type"] == "data_missing"]
        if not data_missing:
            return []
        by_agent = {}
        for i in data_missing:
            a = i["agent"]
            by_agent.setdefault(a, []).append(i)
        return [{
            "target": f"data.py 中 {agent} 使用的数据源",
            "action": f"为 {agent} 添加备用数据源或缓存机制",
            "priority": "medium",
            "reason": f"发现 {len(issues)} 次数据缺失",
        } for agent, issues in by_agent.items()]

    @classmethod
    def _suggest_rating_bias(cls, findings: dict) -> list[dict]:
        rating_bias = [i for i in findings.get("issues", []) if i["type"] == "rating_bias"]
        if not rating_bias:
            return []
        return [{
            "target": "decision.py 决策树阈值",
            "action": "检查评分分布是否合理，考虑调整 z-score 缩放因子",
            "priority": "high",
            "reason": rating_bias[0]["detail"],
        }]

    @classmethod
    def _suggest_agent_underperforming(cls, findings: dict) -> list[dict]:
        underperformers = [i for i in findings.get("issues", []) if i["type"] == "agent_underperforming"]
        if not underperformers:
            return []
        by_agent = {}
        for i in underperformers:
            a = i.get("agent", "")
            by_agent.setdefault(a, []).append(i)
        return [{
            "target": f"agents/{agent_name}",
            "action": f"审查 {agent_name} 的 prompt 和数据输入，准确率低于随机水平可能意味特征失效",
            "priority": "high",
            "reason": f"Agent {agent_name} 方向准确率 < 45%",
        } for agent_name in by_agent]

    @classmethod
    def _suggest_calibration(cls, findings: dict) -> list[dict]:
        suggestions = []
        calibration_conservative = [i for i in findings.get("issues", []) if i["type"] == "calibration_conservative"]
        if calibration_conservative:
            suggestions.append({
                "target": "decision.py 评分映射",
                "action": "高分标的买入操作占比不足，考虑扩大评分范围或调整买入阈值",
                "priority": "medium",
                "reason": calibration_conservative[0]["detail"],
            })
        calibration_aggressive = [i for i in findings.get("issues", []) if i["type"] == "calibration_aggressive"]
        if calibration_aggressive:
            suggestions.append({
                "target": "decision.py 评分映射",
                "action": "低分标的卖出操作占比不足，评分偏激进，考虑收紧评分",
                "priority": "medium",
                "reason": calibration_aggressive[0]["detail"],
            })
        return suggestions

    @classmethod
    def _suggest_agent_redundancy(cls, findings: dict) -> list[dict]:
        redundancies = findings.get("deep_analysis", {}).get("redundancy", [])
        if not redundancies:
            return []
        sorted_red = sorted(redundancies, key=lambda x: x.get("avg_diff", 0))
        return [{
            "target": "agents 配置",
            "action": f"考虑合并或移除{r['agent_a']}与{r['agent_b']}中的一个（平均分差仅{r['avg_diff']}）",
            "priority": "low",
            "reason": "两 Agent 评分高度相关",
        } for r in sorted_red[:5]]

    @classmethod
    def _suggest_llm_override(cls, findings: dict) -> list[dict]:
        llm_excessive = [i for i in findings.get("issues", []) if i["type"] == "llm_override_excessive"]
        if not llm_excessive:
            return []
        return [{
            "target": "决策引擎",
            "action": "增强规则评分系统的特征覆盖，减少对 LLM 判断的依赖",
            "priority": "medium",
            "reason": llm_excessive[0]["detail"],
        }]

    @classmethod
    def _suggest_position_concentration(cls, findings: dict) -> list[dict]:
        concentrated = [i for i in findings.get("issues", []) if i["type"] == "position_too_concentrated"]
        if not concentrated:
            return []
        return [{
            "target": "decision.py 凯利公式",
            "action": "检查 Kelly 公式是否未正确限制最大仓位，当前上限应为 25%",
            "priority": "high",
            "reason": c["detail"],
        } for c in concentrated]

    @classmethod
    def _suggest_total_position(cls, findings: dict) -> list[dict]:
        total_exceed = [i for i in findings.get("issues", []) if i["type"] == "total_position_exceed_100"]
        if not total_exceed:
            return []
        return [{
            "target": "scheduler.py 组合约束",
            "action": "增强 Phase 2.5 归一化逻辑，防止总仓位超过 100%",
            "priority": "high",
            "reason": total_exceed[0]["detail"],
        }]

    @classmethod
    def _suggest_debate_low_impact(cls, findings: dict) -> list[dict]:
        low_debate = [i for i in findings.get("issues", []) if i["type"] == "debate_low_impact"]
        if not low_debate:
            return []
        return [{
            "target": "debate.py",
            "action": "审查辩论流程是否流于形式，考虑引入对抗式辩论或预设分歧议题",
            "priority": "low",
            "reason": low_debate[0]["detail"],
        }]

    @classmethod
    def _suggest_accuracy_declining(cls, findings: dict) -> list[dict]:
        declining = [i for i in findings.get("issues", []) if i["type"] == "accuracy_declining"]
        if not declining:
            return []
        return [{
            "target": "全局系统",
            "action": "系统准确率持续下降，建议全面审查数据源、LLM 模型和特征有效性",
            "priority": "high",
            "reason": declining[0]["detail"],
        }]

    @classmethod
    def _suggest_data_pipeline(cls, findings: dict) -> list[dict]:
        suggestions = []
        trend_issue = [i for i in findings.get("issues", []) if i["type"] == "config_trend_days_too_short"]
        if trend_issue:
            suggestions.append({
                "target": "config.py",
                "action": "增大 TREND_DAY_COUNT 至≥7，过滤短期噪音，提高趋势判断可信度",
                "priority": "medium",
                "reason": trend_issue[0]["detail"],
            })
        fb_issues = [i for i in findings.get("issues", []) if i["type"] in ("high_rule_fallback_rate", "elevated_rule_fallback")]
        if fb_issues:
            suggestions.append({
                "target": "data.py + .env",
                "action": "检查API Key配置和网络连接，添加备用数据源（如akshare新闻）减少Rule Fallback",
                "priority": "high" if fb_issues[0]["type"] == "high_rule_fallback_rate" else "medium",
                "reason": fb_issues[0]["detail"],
            })
        return suggestions

    # ── 新增深度审查维度 ──────────────────────────────────────

    @classmethod
    def _check_data_pipeline(cls, all_reports: list, findings: dict):
        """数据管道健康检查：验证配置参数和数据源链路的合理性。"""
        try:
            from core.config import TREND_DAY_COUNT

            # 1. 舆情趋势天数
            if TREND_DAY_COUNT < 5:
                findings["issues"].append({
                    "type": "config_trend_days_too_short",
                    "severity": "medium",
                    "detail": f"TREND_DAY_COUNT={TREND_DAY_COUNT}<5，趋势判断易受噪音影响，建议≥7",
                })

            # 2. 检查 Rule Fallback 占比（反映数据源健康状况）
            total_agents = 0
            fallback_count = 0
            for fr in all_reports:
                for ar in getattr(fr, "agent_reports", []):
                    total_agents += 1
                    if ar.source == "rule_fallback":
                        fallback_count += 1
            if total_agents > 0:
                fb_pct = fallback_count / total_agents * 100
                if fb_pct > 30:
                    findings["issues"].append({
                        "type": "high_rule_fallback_rate",
                        "severity": "high",
                        "detail": f"Rule Fallback占比{fb_pct:.0f}%（{fallback_count}/{total_agents}），"
                                   f"数据源大面积不可用，请检查API Key和网络连接",
                    })
                elif fb_pct > 15:
                    findings["issues"].append({
                        "type": "elevated_rule_fallback",
                        "severity": "medium",
                        "detail": f"Rule Fallback占比{fb_pct:.0f}%（{fallback_count}/{total_agents}），"
                                   f"部分数据源异常，建议排查",
                    })
        except ImportError:
            logger.warning("导入核心配置失败（TREND_DAY_COUNT）", exc_info=True)
            pass

    @classmethod
    def _check_historical_accuracy(cls, findings: dict):
        """从 ReviewManager 的 cumulative_stats.json 读取历史准确率数据并分析。"""
        if not REVIEW_STATS_FILE.exists():
            return
        try:
            with open(REVIEW_STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            return

        accuracy_data = {"used": False}

        # 1. 整体准确率趋势
        by_date = stats.get("by_date", [])
        if len(by_date) >= 3:
            recent = by_date[-3:]
            recent_avg = np.mean([d["pct"] for d in recent])
            earlier = by_date[:-3]
            earlier_avg = np.mean([d["pct"] for d in earlier]) if earlier else recent_avg
            trend = recent_avg - earlier_avg
            if trend < -5:
                findings["issues"].append({
                    "type": "accuracy_declining",
                    "severity": "high",
                    "detail": f"近3次准确率均值{recent_avg:.1f}%，较之前下降{abs(trend):.1f}个百分点，建议排查",
                })
            elif trend > 5:
                findings["issues"].append({
                    "type": "accuracy_improving",
                    "severity": "low",
                    "detail": f"近3次准确率均值{recent_avg:.1f}%，较之前提升{trend:.1f}个百分点",
                })
            accuracy_data["trend"] = round(trend, 1)
            accuracy_data["recent_avg"] = round(recent_avg, 1)

        # 2. 按 Agent 准确率
        by_agent = stats.get("by_agent", {})
        if by_agent:
            underperformers = []
            for aname, adata in by_agent.items():
                directional = adata.get("directional", 0)
                hits = adata.get("hits", 0)
                if directional >= 5:  # 至少有 5 次方向判断
                    acc = hits / directional * 100
                    if acc < 45:
                        underperformers.append({
                            "agent": aname,
                            "accuracy": round(acc, 1),
                            "samples": directional,
                        })
            if underperformers:
                for u in underperformers:
                    findings["issues"].append({
                        "type": "agent_underperforming",
                        "severity": "high",
                        "agent": u["agent"],
                        "detail": f"{u['agent']}历史方向准确率仅{u['accuracy']}%（样本{u['samples']}次），显著低于随机",
                    })
                accuracy_data["underperformers"] = [u["agent"] for u in underperformers]
            accuracy_data["agent_count"] = len(by_agent)

        # 3. 按操作类型的准确率
        by_op = stats.get("by_operation", {})
        if by_op:
            op_findings = []
            for op, odata in by_op.items():
                total = odata.get("total", 0)
                correct = odata.get("correct", 0)
                if total >= 5:
                    acc = correct / total * 100
                    if op in ("强烈买入", "买入") and acc < 50:
                        op_findings.append(f"{op}准确率仅{acc:.0f}%（{correct}/{total}），买入信号可靠性不足")
                    elif op in ("强烈卖出", "卖出") and acc < 50:
                        op_findings.append(f"{op}准确率仅{acc:.0f}%（{correct}/{total}），卖出信号可靠性不足")
            if op_findings:
                for detail in op_findings:
                    findings["issues"].append({
                        "type": "operation_accuracy_low",
                        "severity": "medium",
                        "detail": detail,
                    })
                accuracy_data["op_issues"] = op_findings

        accuracy_data["used"] = True
        findings.setdefault("deep_analysis", {})["historical_accuracy"] = accuracy_data

    @classmethod
    def _check_calibration(cls, all_reports: list, findings: dict):
        """检查评分区间是否映射到合理的操作/风险水平。"""
        if not all_reports:
            return
        high_scores = [fr for fr in all_reports if fr.final_score >= 70]
        low_scores = [fr for fr in all_reports if fr.final_score < 30]
        findings_list = []

        if len(high_scores) > 3:
            buys = sum(1 for fr in high_scores if fr.operation in ("强烈买入", "买入", "长期持有"))
            buy_pct = buys / len(high_scores) * 100
            if buy_pct < 60:
                findings_list.append({
                    "type": "calibration_conservative",
                    "severity": "medium",
                    "detail": f"高分标的(≥70)仅{buy_pct:.0f}%为买入操作，评分体系可能偏保守",
                })

        if len(low_scores) > 3:
            sells = sum(1 for fr in low_scores if fr.operation in ("减持", "卖出", "强烈卖出"))
            sell_pct = sells / len(low_scores) * 100
            if sell_pct < 60:
                findings_list.append({
                    "type": "calibration_aggressive",
                    "severity": "medium",
                    "detail": f"低分标的(<30)仅{sell_pct:.0f}%为卖出操作，评分体系可能偏激进",
                })

        # 逐档统计
        buckets = {}
        for fr in all_reports:
            bucket = (int(fr.final_score) // 10) * 10
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "buys": 0, "sells": 0, "holds": 0}
            buckets[bucket]["count"] += 1
            if fr.operation in ("强烈买入", "买入", "长期持有"):
                buckets[bucket]["buys"] += 1
            elif fr.operation in ("减持", "卖出", "强烈卖出"):
                buckets[bucket]["sells"] += 1
            else:
                buckets[bucket]["holds"] += 1

        findings.setdefault("deep_analysis", {})["calibration"] = {
            "buckets": {str(k): v for k, v in sorted(buckets.items())},
            "issues": [f["detail"] for f in findings_list],
        }
        for f in findings_list:
            findings["issues"].append(f)

    @classmethod
    def _check_agent_redundancy(cls, all_reports: list, findings: dict):
        """检测持续产出相似评分的智能体对。"""
        if not all_reports:
            return
        agent_scores = defaultdict(list)
        for fr in all_reports:
            for ar in fr.agent_reports:
                agent_scores[ar.agent_name].append(ar.score)

        redundancies = []
        names = list(agent_scores.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_scores = agent_scores[names[i]]
                b_scores = agent_scores[names[j]]
                if len(a_scores) < 5 or len(b_scores) < 5:
                    continue
                avg_diff = np.mean([abs(a - b) for a, b in zip(a_scores, b_scores)])
                if avg_diff < 8:
                    redundancies.append({
                        "agent_a": names[i],
                        "agent_b": names[j],
                        "avg_diff": round(avg_diff, 1),
                    })

        if redundancies:
            for r in redundancies:
                findings["issues"].append({
                    "type": "agent_redundant",
                    "severity": "low",
                    "detail": f"{r['agent_a']}与{r['agent_b']}平均分差仅{r['avg_diff']}，高度冗余",
                })
            findings.setdefault("deep_analysis", {})["redundancy"] = redundancies

    @classmethod
    def _check_debate_impact(cls, all_reports: list, findings: dict):
        """检查辩论是否对最终决策产生了实质性影响。"""
        debate_count = sum(1 for fr in all_reports if fr.debates)
        if debate_count == 0:
            return

        # 检查辩论文本中是否包含实质性修改信号
        meaningful_debates = 0
        for fr in all_reports:
            if not fr.debates:
                continue
            for debate in fr.debates:
                debate_text = str(debate)
                if any(kw in debate_text for kw in ("调整", "改分", "修正", "重新评估", "改为", "提分", "降分")):
                    meaningful_debates += 1
                    break

        if debate_count > 0:
            impact_pct = meaningful_debates / debate_count * 100
            if impact_pct < 30 and debate_count >= 3:
                findings["issues"].append({
                    "type": "debate_low_impact",
                    "severity": "low",
                    "detail": f"仅{impact_pct:.0f}%的辩论涉及实质性评分调整（{meaningful_debates}/{debate_count}），辩论可能流于形式",
                })

            findings.setdefault("deep_analysis", {})["debate_impact"] = {
                "total_with_debates": debate_count,
                "meaningful_debates": meaningful_debates,
                "impact_pct": round(impact_pct, 1),
            }

    @classmethod
    def _check_llm_override_rate(cls, all_reports: list, findings: dict):
        """跟踪 LLM 覆盖规则评分的比例。"""
        if not all_reports:
            return
        llm_override = 0
        rule_based = 0
        for fr in all_reports:
            if "【规则综合】" in fr.core_logic or "【规则评分】" in fr.core_logic:
                rule_based += 1
            else:
                llm_override += 1

        total = llm_override + rule_based
        if total == 0:
            return

        override_pct = llm_override / total * 100
        if override_pct > 60:
            findings["issues"].append({
                "type": "llm_override_excessive",
                "severity": "medium",
                "detail": f"LLM覆盖规则占比{override_pct:.0f}%（{llm_override}/{total}），规则系统可能需要增强",
            })
        elif override_pct < 10:
            findings["issues"].append({
                "type": "llm_override_too_low",
                "severity": "low",
                "detail": f"LLM几乎从不覆盖规则评分（{override_pct:.0f}%），LLM可能过于保守",
            })

        findings.setdefault("deep_analysis", {})["llm_override"] = {
            "llm_count": llm_override,
            "rule_count": rule_based,
            "override_pct": round(override_pct, 1),
        }

    @classmethod
    def _check_position_sizing(cls, all_reports: list, findings: dict):
        """验证凯利公式输出的仓位是否合理。"""
        if not all_reports:
            return
        issues = []

        # 单只 ETF 仓位异常
        for fr in all_reports:
            pos = fr.suggested_position_pct
            if pos > 0.25:
                issues.append({
                    "type": "position_too_concentrated",
                    "severity": "high",
                    "detail": f"{fr.etf_info['name']}({fr.etf_info['code']})仓位{pos*100:.0f}% > 25%，集中度风险",
                })
            if 0 < pos < 0.005:
                issues.append({
                    "type": "position_too_small",
                    "severity": "low",
                    "detail": f"{fr.etf_info['name']}({fr.etf_info['code']})仓位{pos*100:.2f}%过小，无可操作性",
                })

        # 总仓位检查
        total_pos = sum(fr.suggested_position_pct for fr in all_reports)
        if total_pos > 1.0:
            issues.append({
                "type": "total_position_exceed_100",
                "severity": "high",
                "detail": f"总仓位{total_pos*100:.0f}% > 100%，异常（应已被scheduler归一化）",
            })

        if issues:
            for issue in issues:
                findings["issues"].append(issue)
            findings.setdefault("deep_analysis", {})["position_sizing"] = {
                "issues": [i["detail"] for i in issues],
                "total_position_pct": round(total_pos * 100, 1),
            }

    @classmethod
    def _save(cls, findings: dict):
        """保存（累积）审查结果。"""
        os.makedirs(os.path.dirname(OPTIMIZATION_FILE), exist_ok=True)
        
        history = []
        if OPTIMIZATION_FILE.exists():
            try:
                with open(OPTIMIZATION_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                logger.warning("加载优化历史失败", exc_info=True)
                pass
        
        # 最多保留 30 天记录
        history.append(findings)
        if len(history) > 30:
            history = history[-30:]
        
        with open(OPTIMIZATION_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_pending_suggestions(cls) -> list[dict]:
        """获取最近一次运行中尚未解决的优化建议。"""
        if not OPTIMIZATION_FILE.exists():
            return []
        try:
            with open(OPTIMIZATION_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                return []
            latest = history[-1]
            return latest.get("suggestions", [])
        except Exception:
            return []

    @classmethod
    def print_summary(cls, findings: dict):
        """打印审查摘要（含深度分析维度）。"""
        issues = findings.get("issues", [])
        suggestions = findings.get("suggestions", [])
        deep = findings.get("deep_analysis", {})

        print(f"\n{'='*60}")
        print(f"  [DIRECTOR] 工作流总监审查报告")
        print(f"{'='*60}")
        print(f"  审查日期: {findings['date']}")
        print(f"  分析标的: {findings['total_etfs']} 只 ETF")

        # ── 全景问题统计 ──
        if issues:
            by_severity = {"high": 0, "medium": 0, "low": 0}
            by_type = {}
            for i in issues:
                by_severity[i["severity"]] = by_severity.get(i["severity"], 0) + 1
                t = i["type"]
                by_type[t] = by_type.get(t, 0) + 1

            print(f"\n  发现问题: {len(issues)} 个")
            print(f"    高: {by_severity['high']} | 中: {by_severity['medium']} | 低: {by_severity['low']}")
            print(f"    类型分布: {dict(sorted(by_type.items(), key=lambda x: -x[1])[:6])}")

            # 打印高优先级问题
            high_issues = [i for i in issues if i["severity"] == "high"][:5]
            for i in high_issues:
                print(f"    [HIGH] {i['detail'][:90]}")
        else:
            print(f"\n  未发现问题")

        # ── 深度分析摘要 ──
        if deep:
            print(f"\n  [深度分析]")
            # 历史准确率
            ha = deep.get("historical_accuracy", {})
            if ha.get("used"):
                parts = []
                if "recent_avg" in ha:
                    parts.append(f"近3次准确率{ha['recent_avg']}%")
                if "trend" in ha:
                    arrow = "up" if ha["trend"] > 0 else "down"
                    parts.append(f"趋势{arrow}{abs(ha['trend'])}%")
                if "underperformers" in ha:
                    parts.append(f"低准确率Agent: {len(ha['underperformers'])}个")
                if parts:
                    print(f"    准确率: {' | '.join(parts)}")

            # 校准
            cal = deep.get("calibration", {})
            if cal.get("issues"):
                print(f"    评分校准: {'; '.join(cal['issues'][:2])}")

            # 冗余
            red = deep.get("redundancy", [])
            if red:
                print(f"    Agent冗余: {len(red)}对高度相关")

            # LLM 覆盖
            llm = deep.get("llm_override", {})
            if llm:
                total_llm_check = llm.get('llm_count', 0) + llm.get('rule_count', 0)
                if total_llm_check > 0:
                    print(f"    LLM覆盖规则: {llm['override_pct']}% ({llm['llm_count']}/{total_llm_check})")

            # 仓位
            ps = deep.get("position_sizing", {})
            if ps:
                pi = ps.get("issues", [])
                if pi:
                    print(f"    仓位异常: {len(pi)}个问题 | 总仓位{ps.get('total_position_pct', 0)}%")

        if suggestions:
            print(f"\n  优化建议: {len(suggestions)} 条")
            by_pri = {"high": [], "medium": [], "low": []}
            for s in suggestions:
                by_pri[s["priority"]].append(s)
            for pri in ("high", "medium", "low"):
                if by_pri[pri]:
                    print(f"    [{pri.upper()}]")
                    for s in by_pri[pri][:2]:
                        print(f"      → {s['action'][:70]}")
        print(f"{'='*60}\n")

