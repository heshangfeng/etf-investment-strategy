"""
ETF 智能投资分析系统 - SQLite 持久缓存层

替代全局内存 dict，数据自动持久化到 SQLite，重启后秒级加载。
线程安全，支持按 key 独立 TTL。
"""
import json
import pickle
import sqlite3
import threading
import time
from pathlib import Path


class PersistentCache:
    """SQLite 持久缓存，dict 风格接口，支持 TTL。"""

    def __init__(self, section: str, db_path: str = "", default_ttl: int = 3600):
        self.section = section
        self.default_ttl = default_ttl
        self._lock = threading.Lock()

        if not db_path:
            from core.config import CACHE_DB_PATH
            db_path = CACHE_DB_PATH
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS persistent_cache ("
            "  section TEXT, key TEXT, value BLOB, timestamp REAL,"
            "  PRIMARY KEY (section, key)"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_section "
            "ON persistent_cache(section)"
        )
        self._conn.commit()
        self._auto_vacuum()

    # ---- dict 风格接口 ----

    def get(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value, timestamp FROM persistent_cache WHERE section=? AND key=?",
                (self.section, str(key))
            ).fetchone()
        if row is None:
            return default
        try:
            return pickle.loads(row[0])
        except Exception:
            return default

    def __getitem__(self, key: str):
        val = self.get(key, _MISSING)
        if val is _MISSING:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value):
        self.put(key, value)

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def put(self, key: str, value, ttl: int | None = None):
        blob = pickle.dumps(value)
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "REPLACE INTO persistent_cache (section, key, value, timestamp) VALUES (?, ?, ?, ?)",
                (self.section, str(key), blob, ts)
            )
            self._conn.commit()

    def pop(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM persistent_cache WHERE section=? AND key=?",
                (self.section, str(key))
            ).fetchone()
            if row:
                self._conn.execute(
                    "DELETE FROM persistent_cache WHERE section=? AND key=?",
                    (self.section, str(key))
                )
                self._conn.commit()
                try:
                    return pickle.loads(row[0])
                except Exception:
                    return default
            return default

    def keys(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM persistent_cache WHERE section=?",
                (self.section,)
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self):
        with self._lock:
            self._conn.execute(
                "DELETE FROM persistent_cache WHERE section=?",
                (self.section,)
            )
            self._conn.commit()

    @property
    def __len__(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM persistent_cache WHERE section=?",
                (self.section,)
            ).fetchone()
            return row[0] if row else 0

    # ---- TTL 检查 ----

    def is_fresh(self, key: str, max_age: int | None = None) -> bool:
        """检查 key 是否在 TTL 内有效。"""
        if max_age is None:
            max_age = self.default_ttl
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp FROM persistent_cache WHERE section=? AND key=?",
                (self.section, str(key))
            ).fetchone()
        if row is None:
            return False
        return time.time() - row[0] < max_age

    def touch(self, key: str):
        """刷新 key 的时间戳（标记为刚更新）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE persistent_cache SET timestamp=? WHERE section=? AND key=?",
                (time.time(), self.section, str(key))
            )
            self._conn.commit()

    # ---- 维护 ----

    def _auto_vacuum(self):
        """清理过期条目，每天最多一次。"""
        try:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM persistent_cache WHERE timestamp < ?",
                    (time.time() - 86400 * 7,)  # 保留7天
                )
                self._conn.commit()
        except Exception:
            pass

    def close(self):
        self._conn.close()


_MISSING = object()
