"""
ETF 智能投资分析系统 - 工作流总监 Agent

每次运行后自动审查各 Agent 输出质量，发现盲点，累积优化建议。
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR, REVIEW_DIR

OPTIMIZATION_FILE = Path(REVIEW_DIR) / "optimization.json"


class WorkflowDirector:
    """
    工作流总监：审查 Agent 输出 → 发现盲点 → 累积优化建议 → 反馈到下次运行。
    
    审查维度：
    1. 数据完整性：Agent 是否拿到了足够的数据
    2. 输出质量：分析是否实质性还是空泛
    3. 覆盖盲点：哪些维度没有 Agent 覆盖
    4. 自相矛盾：Agent 的断言是否有数据支持
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

        # ── 4. 生成优化建议 ──
        cls._generate_suggestions(findings)

        # ── 5. 保存结果（累积） ──
        cls._save(findings)

        return findings

    @classmethod
    def _generate_suggestions(cls, findings: dict):
        """根据发现的问题生成优化建议。"""
        suggestions = []
        
        # Agent 缺失 → 检查是代码问题还是 LLM 调用失败
        agent_missing = [i for i in findings.get("issues", []) if i["type"] == "agent_missing"]
        if agent_missing:
            suggestions.append({
                "target": "scheduler.py",
                "action": "检查 Agent 并行执行是否有超时或异常退出",
                "priority": "high",
                "reason": f"发现 {len(agent_missing)} 只 ETF 的 Agent 数量不足",
            })

        # 空泛分析 → 可能需要优化 prompt
        generic = [i for i in findings.get("issues", []) if i["type"] == "generic_analysis"]
        if generic:
            # 按 Agent 分组
            by_agent = {}
            for i in generic:
                a = i["agent"]
                by_agent.setdefault(a, []).append(i)
            for agent, issues in by_agent.items():
                suggestions.append({
                    "target": f"agents 中的 {agent}",
                    "action": f"增强 {agent} 的 SYSTEM_PROMPT，要求输出更详细的分析（至少 100 字）",
                    "priority": "medium",
                    "reason": f"发现 {len(issues)} 次空泛分析",
                })

        # 数据缺失 → 检查数据源
        data_missing = [i for i in findings.get("issues", []) if i["type"] == "data_missing"]
        if data_missing:
            by_agent = {}
            for i in data_missing:
                a = i["agent"]
                by_agent.setdefault(a, []).append(i)
            for agent, issues in by_agent.items():
                suggestions.append({
                    "target": f"data.py 中 {agent} 使用的数据源",
                    "action": f"为 {agent} 添加备用数据源或缓存机制",
                    "priority": "medium",
                    "reason": f"发现 {len(issues)} 次数据缺失",
                })

        # 评级偏斜
        rating_bias = [i for i in findings.get("issues", []) if i["type"] == "rating_bias"]
        if rating_bias:
            suggestions.append({
                "target": "decision.py 决策树阈值",
                "action": "检查评分分布是否合理，考虑调整 z-score 缩放因子",
                "priority": "high",
                "reason": rating_bias[0]["detail"],
            })

        findings["suggestions"] = suggestions

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
        """打印审查摘要。"""
        issues = findings.get("issues", [])
        suggestions = findings.get("suggestions", [])

        print(f"\n{'='*60}")
        print(f"  🔍 工作流总监审查报告")
        print(f"{'='*60}")
        print(f"  审查日期: {findings['date']}")
        print(f"  分析标的: {findings['total_etfs']} 只 ETF")

        if issues:
            by_severity = {"high": 0, "medium": 0, "low": 0}
            for i in issues:
                by_severity[i["severity"]] = by_severity.get(i["severity"], 0) + 1
            print(f"\n  发现问题: {len(issues)} 个")
            print(f"    高: {by_severity['high']} | 中: {by_severity['medium']} | 低: {by_severity['low']}")
            
            # 打印高优先级问题
            high_issues = [i for i in issues if i["severity"] == "high"][:3]
            for i in high_issues:
                print(f"    🔴 {i['detail'][:80]}")
        else:
            print(f"\n  未发现问题 ✅")

        if suggestions:
            print(f"\n  优化建议: {len(suggestions)} 条")
            for s in suggestions[:3]:
                print(f"    → [{s['priority']}] {s['action'][:60]}")
        print(f"{'='*60}\n")
