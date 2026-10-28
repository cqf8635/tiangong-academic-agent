"""TianGong Academic Radar Daily Sync & Persistent Snapshot Engine.

Ensures:
1. Whenever backend starts up (even after several days offline), immediately checks and refreshes today's latest papers.
2. If running 24/7, continuously auto-syncs every 6 hours and especially at arXiv daily publication window (08:30-09:30 AM Beijing Time).
3. Persists local JSON snapshots to disk (data/radar_cache/), enabling instant offline/zero-latency startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent
RADAR_CACHE_DIR = BASE_DIR / "data" / "radar_cache"
RADAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 关注的每日核心学科大类列表
CORE_MAJORS = ["PHYSICS", "CS", "MATH", "Q-BIO", "Q-FIN", "STAT", "EESS", "ECON"]

# 经典重点追踪子学科
CORE_SUB_CATEGORIES = {
    "PHYSICS": ["astro-ph.HE", "cond-mat.supr-con", "quant-ph", "hep-th"],
    "CS": ["cs.AI", "cs.RO", "cs.LG", "cs.CL", "cs.CR"],
    "MATH": ["math.OC", "math.AP", "math.PR"],
    "Q-BIO": ["q-bio.BM", "q-bio.NC", "q-bio.GN"],
    "Q-FIN": ["q-fin.PM", "q-fin.RM", "q-fin.TR", "q-fin.MF"],
    "STAT": ["stat.ML", "stat.TH", "stat.ME"],
    "EESS": ["eess.SP", "eess.IV", "eess.SY"],
    "ECON": ["econ.EM", "econ.TH", "econ.GN"],
}


class RadarSyncEngine:
    def __init__(self, paper_searcher: Any):
        self.paper_searcher = paper_searcher
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._last_sync_timestamp: float = 0.0

    def get_cache_file(self, major: str, sub_key: str = "all") -> Path:
        safe_sub = sub_key.replace(".", "_").replace("*", "_all")
        return RADAR_CACHE_DIR / f"radar_{major}_{safe_sub}.json"

    def read_snapshot(self, major: str, sub_key: str = "all") -> Optional[Dict[str, Any]]:
        """从本地磁盘读取持久化快照，做到秒开且永不断档"""
        cache_f = self.get_cache_file(major, sub_key)
        if cache_f.exists():
            try:
                with open(cache_f, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                logger.warning(f"读取雷达本地快照失败 ({cache_f}): {e}")
        return None

    def save_snapshot(self, major: str, sub_key: str, data: Dict[str, Any]):
        """持久化保存最新雷达检索快照到本地"""
        cache_f = self.get_cache_file(major, sub_key)
        try:
            with open(cache_f, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入雷达快照失败 ({cache_f}): {e}")

    def sync_core_majors_now(self):
        """执行一次全量核心大类与重点子学科同步 (具备智能新鲜度检测与合规防限流机制)"""
        logger.info("[天工雷达] 开始检测最新各学科预印本并刷新本地快照...")

        now = time.time()
        for major in CORE_MAJORS:
            if self._stop_event.is_set():
                break

            # 避让检测：若检测到前台用户在活跃检索，后台同步主动挂起让行
            while getattr(self.paper_searcher, "is_user_active", lambda w=30.0: False)(30.0) and not self._stop_event.is_set():
                logger.info("[天工雷达] 检测到前台用户正在检索交互，后台同步主动礼让避让 15 秒...")
                time.sleep(15)

            try:
                # 1. 检查大类快照是否在 8 小时内已存在且新鲜，若新鲜则跳过避免洪峰
                snap_all = self.get_cache_file(major, "all")
                if not snap_all.exists() or (now - snap_all.stat().st_mtime > 8 * 3600):
                    res = self.paper_searcher.search_radar(
                        major=major,
                        sort_by="submittedDate",
                        max_results=35,
                        is_background=True,
                    )
                    if res and res.get("papers") and not res.get("is_cached_snapshot"):
                        res["snapshot_synced_at"] = datetime.now().isoformat()
                        self.save_snapshot(major, "all", res)
                        logger.info(f"[天工雷达] 学科 [{major}] 同步成功: 捕获 {len(res['papers'])} 卷最新论著")
                    time.sleep(4.0)  # 严格遵循 arXiv 合规间隔

                # 2. 重点关键子学科同步（如天体物理 astro-ph.HE）
                for sub in CORE_SUB_CATEGORIES.get(major, []):
                    if self._stop_event.is_set():
                        break

                    while getattr(self.paper_searcher, "is_user_active", lambda w=30.0: False)(30.0) and not self._stop_event.is_set():
                        logger.info("[天工雷达] 检测到前台用户正在检索交互，后台同步主动礼让避让 15 秒...")
                        time.sleep(15)

                    snap_sub = self.get_cache_file(major, sub)
                    if not snap_sub.exists() or (now - snap_sub.stat().st_mtime > 8 * 3600):
                        sub_res = self.paper_searcher.search_radar(
                            major=major,
                            sub_categories=[sub],
                            sort_by="submittedDate",
                            max_results=25,
                            is_background=True,
                        )
                        if sub_res and sub_res.get("papers") and not sub_res.get("is_cached_snapshot"):
                            sub_res["snapshot_synced_at"] = datetime.now().isoformat()
                            self.save_snapshot(major, sub, sub_res)
                        time.sleep(4.0)  # 严格遵循 arXiv 合规间隔
            except Exception as e:
                logger.warning(f"[天工雷达] 学科 [{major}] 同步受阻: {e}")

        self._last_sync_timestamp = time.time()
        logger.info("[天工雷达] 每日文献同步完成，本地快照已更新完毕")

    def _background_loop(self):
        """24小时常驻轮询调度器：无论常驻还是隔几天启动，都自动保持最新"""
        # 1. 启动即检查：如果从没同步过或本地快照已超过 6 小时，延后 30 秒热同步一次（给系统初始化与前台访问让路）
        now = time.time()
        need_immediate_sync = False
        check_cache = self.get_cache_file("PHYSICS", "all")
        if not check_cache.exists():
            need_immediate_sync = True
        else:
            mtime = check_cache.stat().st_mtime
            if now - mtime > 6 * 3600:  # 超过6小时
                need_immediate_sync = True

        if need_immediate_sync:
            # 延时 30 秒，确保后端所有进程完全就绪，且优先让前台用户完成首次访问
            for _ in range(6):
                if self._stop_event.is_set():
                    break
                time.sleep(5)
            if not self._stop_event.is_set():
                self.sync_core_majors_now()

        # 2. 常驻循环：每 6 小时巡检一次
        sync_interval_seconds = 6 * 3600
        while not self._stop_event.is_set():
            # 采用 10 秒递进检测，便于优雅退出
            for _ in range(int(sync_interval_seconds // 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(10)

            if not self._stop_event.is_set():
                self.sync_core_majors_now()

    def start_scheduler(self):
        """启动后台自动更新线程"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="TianGongRadarSyncThread",
        )
        self._worker_thread.start()

    def stop_scheduler(self):
        """停止后台同步线程"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
