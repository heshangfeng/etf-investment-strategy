"""
ETF 智能投资分析系统 - BM25 轻量记忆系统
从历史快照中检索同ETF的过往分析和复盘结果。
"""
import json
import glob
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


class SimpleBM25:
    """标准的 BM25 检索实现。"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0

    def fit(self, documents: list[str]):
        """Build BM25 index from document strings."""
        self.documents = documents
        tokenized = [doc.split() for doc in documents]
        self.doc_freqs = [Counter(tokens) for tokens in tokenized]

        # Compute IDF
        n_docs = len(documents)
        all_terms = set()
        for freq in self.doc_freqs:
            all_terms.update(freq.keys())

        for term in all_terms:
            df = sum(1 for freq in self.doc_freqs if term in freq)
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        self.avg_doc_len = sum(sum(freq.values()) for freq in self.doc_freqs) / max(n_docs, 1)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for one query-doc pair."""
        query_terms = query.split()
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = sum(doc_freq.values())

        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = doc_freq.get(term, 0)
            idf = self.idf[term]
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1)))
        return score


class MemoryRetriever:
    """
    从历史快照中检索同ETF的过往分析记录。
    每次检索返回最近N天的分析摘要+复盘结果。
    """

    SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

    @classmethod
    def retrieve(cls, etf_code: str, days: int = 20, top_k: int = 5) -> str:
        """
        检索指定ETF的历史分析记录。

        Returns:
            格式化的历史分析文本（空字符串表示无历史记录）
        """
        snapshots = cls._load_snapshots(days)
        if not snapshots:
            return ""

        records = []
        for snap in snapshots:
            for etf in snap.get("etfs", []):
                if etf.get("code") == etf_code:
                    agents = etf.get("agents", [])
                    for a in agents:
                        records.append({
                            "date": snap.get("date", ""),
                            "agent": a.get("n", ""),
                            "rating": a.get("rt", ""),
                            "score": a.get("sc", 0),
                            "source": a.get("src", ""),
                        })
                    # Also add the final decision
                    records.append({
                        "date": snap.get("date", ""),
                        "agent": "首席决策",
                        "rating": etf.get("final_rating", ""),
                        "score": etf.get("final_score", 0),
                        "source": "chief",
                    })
                    break

        if not records:
            return ""

        # Sort by date desc, keep top_k
        records.sort(key=lambda r: r["date"], reverse=True)
        records = records[:top_k]

        lines = ["【历史分析记录（最近{}天）】".format(days)]
        for r in records:
            lines.append(f"  {r['date']} | {r['agent']:10s} | {r['rating']} ({r['score']})")

        return "\n".join(lines)

    @classmethod
    def _load_snapshots(cls, days: int) -> list[dict]:
        """加载最近N天的快照。"""
        cutoff = datetime.now() - timedelta(days=days)
        snaps = []
        for fpath in sorted(glob.glob(str(cls.SNAPSHOT_DIR / "*.json"))):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    snap = json.load(f)
                snap_date = snap.get("date", "")
                try:
                    dt = datetime.strptime(snap_date, "%Y%m%d")
                    if dt >= cutoff:
                        snaps.append(snap)
                except:
                    snaps.append(snap)
            except:
                continue
        return snaps
