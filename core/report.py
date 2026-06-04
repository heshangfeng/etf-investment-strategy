"""
ETF 智能投资分析系统 - 投研报告生成器
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.config import RATING_ORDER
from core.models import FinalResearchReport


class ResearchReportGenerator:
    """生成完整的ETF多智能体投研报告——不止表格，含每Agent分析原文+辩论+首席决策"""

    @staticmethod
    def generate_full_report(all_reports: list[FinalResearchReport]):
        today = datetime.now().strftime("%Y-%m-%d")
        print("\n" + "█"*160)
        print(f"  ETF 多智能体投研报告 | {today}")
        print("  LLM多角色专家分析 + 矛盾检测 + 选择性辩论 + 首席综合决策")
        print("█"*160)

        df_summary = ResearchReportGenerator._print_summary_board(all_reports)
        ResearchReportGenerator._print_operation_tiers(all_reports)
        ResearchReportGenerator._print_detailed_reports(all_reports)
        ResearchReportGenerator._print_sector_rotation(all_reports)
        ResearchReportGenerator._print_portfolio_summary(all_reports)
        ResearchReportGenerator._print_risk_sections(all_reports)

        from core.config import OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = f"{OUTPUT_DIR}/ETF_多智能体投研报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df_summary.to_excel(save_path, index=False)
        print(f"\n✅ 投研摘要已保存：{save_path}")

        txt_path = f"{OUTPUT_DIR}/ETF_多智能体投研报告_{datetime.now().strftime('%Y%m%d')}.txt"
        full_text = ResearchReportGenerator._build_full_report_text(all_reports)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✅ 完整投研报告已保存：{txt_path}")

        ResearchReportGenerator._cleanup_old_reports()

    @staticmethod
    def _print_summary_board(all_reports: list[FinalResearchReport]) -> pd.DataFrame:
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
        return df_summary

    @staticmethod
    def _print_operation_tiers(all_reports: list[FinalResearchReport]):
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

    @staticmethod
    def _print_detailed_reports(all_reports: list[FinalResearchReport]):
        print("\n" + "="*160)
        print("【📋 各标的多智能体详细报告】")
        print("="*160)

        for fr in all_reports:
            ResearchReportGenerator._print_detailed_report(fr)

    @staticmethod
    def _print_sector_rotation(all_reports: list[FinalResearchReport]):
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

    @staticmethod
    def _print_portfolio_summary(all_reports: list[FinalResearchReport]):
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
        buy_ratio = buy_cnt / total * 100 if total > 0 else 0
        sell_ratio = sell_cnt / total * 100 if total > 0 else 0
        print(f"  多头方向: {buy_cnt}只 ({buy_ratio:.0f}%) | 空头方向: {sell_cnt}只 ({sell_ratio:.0f}%)")
        if buy_ratio > 70:
            print(f"  ⚠️ 集中度预警: 超过{70}%标的集中在多头方向，注意一致性风险")
        if sell_ratio > 40:
            print(f"  ⚠️ 集中度预警: 超过{40}%标的集中在空头方向，市场情绪过度悲观")

    @staticmethod
    def _print_risk_sections(all_reports: list[FinalResearchReport]):
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

    @staticmethod
    def _cleanup_old_reports():
        """保留当天最新报告，删除同一日期的旧版本（带时间戳的旧格式）。"""
        from core.config import OUTPUT_DIR
        today = datetime.now().strftime("%Y%m%d")
        removed = 0
        for fname in list(os.listdir(OUTPUT_DIR)):
            if "ETF_多智能体投研报告_" not in fname:
                continue
            if today in fname and fname.count("_") >= 3:
                os.remove(os.path.join(OUTPUT_DIR, fname))
                removed += 1
        if removed:
            print(f"  🧹 已清理 {removed} 份旧格式报告（已覆盖）")

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

        lines.append("\n【投研摘要看板】")
        lines.append(sep)
        for fr in all_reports:
            lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} ({fr.etf_info['type']}) | {fr.operation} | 仓位:{fr.suggested_position_pct*100:.0f}% | 周期:{fr.holding_period} | 共识:{fr.consensus_level}")

        lines.append(ResearchReportGenerator._build_tier_summary(all_reports))

        lines.append(f"\n{sep}")
        lines.append("【各标的多智能体详细报告】")
        lines.append(sep)

        for fr in all_reports:
            lines.append(ResearchReportGenerator._build_single_etf_report(fr))

        lines.append(ResearchReportGenerator._build_sector_rotation_text(all_reports))
        lines.append(ResearchReportGenerator._build_portfolio_summary_text(all_reports))
        lines.append(ResearchReportGenerator._build_risk_section_text(all_reports))

        return "\n".join(lines)

    @staticmethod
    def _build_tier_summary(all_reports: list[FinalResearchReport]) -> str:
        lines = []
        strong_buy = [fr for fr in all_reports if fr.operation == "强烈买入"]
        buy = [fr for fr in all_reports if fr.operation == "买入"]
        long_hold = [fr for fr in all_reports if fr.operation == "长期持有"]
        hold = [fr for fr in all_reports if fr.operation == "持有"]
        reduce = [fr for fr in all_reports if fr.operation == "减持"]
        sell = [fr for fr in all_reports if fr.operation in ("卖出", "强烈卖出")]
        if strong_buy:
            lines.append(f"\n【强烈买入（{len(strong_buy)}只）】")
            for fr in strong_buy:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        if buy:
            lines.append(f"\n【买入（{len(buy)}只）】")
            for fr in buy:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 仓位:{fr.suggested_position_pct*100:.0f}% | {fr.core_logic[:100]}")
        if long_hold:
            lines.append(f"\n【长期持有（{len(long_hold)}只）】")
            for fr in long_hold:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 周期:{fr.holding_period} | {fr.core_logic[:80]}")
        if hold:
            lines.append(f"\n【持有（{len(hold)}只）】")
            for fr in hold:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | 共识:{fr.consensus_level}")
        if reduce:
            lines.append(f"\n【减持（{len(reduce)}只）】")
            for fr in reduce:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        if sell:
            lines.append(f"\n【卖出/强烈卖出（{len(sell)}只）】")
            for fr in sell:
                lines.append(f"  {fr.etf_info['code']} {fr.etf_info['name']} | {fr.risk_summary[:80]}")
        return "\n".join(lines)

    @staticmethod
    def _build_single_etf_report(fr: FinalResearchReport) -> str:
        lines = []
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
                lines.append(f"  {d.get('topic', d.get('focus', '分歧'))}")
                if 'winner' in d:
                    lines.append(f"  胜方: {d.get('winner', '折中')} | 裁决: {d.get('reasoning', '')[:200]}")
                elif 'rounds' in d:
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

        return "\n".join(lines)

    @staticmethod
    def _build_sector_rotation_text(all_reports: list[FinalResearchReport]) -> str:
        lines = []
        sep = "=" * 160

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

        return "\n".join(lines)

    @staticmethod
    def _build_portfolio_summary_text(all_reports: list[FinalResearchReport]) -> str:
        lines = []
        sep = "=" * 160

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
        buy_ratio = buy_cnt / total * 100 if total > 0 else 0
        sell_ratio = sell_cnt / total * 100 if total > 0 else 0
        lines.append(f"  多头方向: {buy_cnt}只 ({buy_ratio:.0f}%) | 空头方向: {sell_cnt}只 ({sell_ratio:.0f}%)")
        if buy_ratio > 70:
            lines.append(f"  ⚠️ 集中度预警: 超过{70}%标的集中在多头方向，注意一致性风险")
        if sell_ratio > 40:
            lines.append(f"  ⚠️ 集中度预警: 超过{40}%标的集中在空头方向，市场情绪过度悲观")

        return "\n".join(lines)

    @staticmethod
    def _build_risk_section_text(all_reports: list[FinalResearchReport]) -> str:
        lines = []
        sep = "=" * 160

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
                print(f"    🎯 {d.get('topic', d.get('focus', '分歧'))}")
                if 'winner' in d:
                    print(f"      胜方: {d.get('winner', '折中')}")
                    print(f"      裁决: {d.get('reasoning', '')[:200]}")
                    if d.get('score_adjustment'):
                        print(f"      调整: {d['score_adjustment']:+.0f}分")
                elif 'rounds' in d:
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
