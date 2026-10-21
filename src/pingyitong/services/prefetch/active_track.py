"""活跃字幕轨注册表（联网预翻译 P2，自研）。

对应方案 v1.2 §3.5（R6）。职责：管理多条已解析字幕轨的生命周期，
维护**唯一活跃轨**（fuzzy 通道的匹配目标），提供启发式推荐与连播提示。

**推荐只建议、不静默切换**（D4）：活跃轨切换改变屏幕上所有字幕的
匹配目标，静默切换一旦选错，用户看到的是整屏错配译文。自动连播
（auto_advance）默认关闭，开启时也只在文件名序列严格连续时才切。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from ...utils.logger import get_logger

logger = get_logger(__name__)

#: 剧集标记：S01E02 / S1E2 / 第 3 集 / EP3 / CD2 等
_EPISODE_PATTERNS = (
    re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})"),
    re.compile(r"第\s*(\d{1,3})\s*[集话]"),
    re.compile(r"(?:EP|ep|CD|cd)[-\s]?(\d{1,3})"),
)


@dataclass
class SubtitleTrack:
    """一条已注册的字幕轨。"""

    track_id: str
    path: str
    title: str = ""
    line_count: int = 0
    lang: str = ""


def _episode_of(name: str) -> tuple[int, ...] | None:
    """提取文件名里的剧集号元组；无标记返回 None。"""
    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(name)
        if match:
            return tuple(int(g) for g in match.groups())
    return None


def next_episode_file(current_video: str, candidates: list[str]) -> str | None:
    """在同目录候选里找"当前视频的下一集"（文件名序列严格连续）。

    严格连续：剧集号必须恰好 +1，其余部分忽略——宁可返回 None
    提示用户手动选，也不能凭相似度猜错。
    """
    current_ep = _episode_of(current_video)
    if current_ep is None:
        return None
    best: tuple[int, ...] | None = None
    best_name = None
    for candidate in candidates:
        episode = _episode_of(candidate)
        if episode is None:
            continue
        if episode[:-1] != current_ep[:-1]:
            continue  # 季/集前缀不同，不算连播
        if episode[-1] == current_ep[-1] + 1 and (best is None or episode[-1] < best[-1]):
            best = episode
            best_name = candidate
    return best_name


def similar_title_score(subtitle_name: str, video_name: str) -> float:
    """字幕文件名与视频文件名的相似度（0~1，去扩展名与剧集号后比较）。"""
    from rapidfuzz import fuzz

    def _clean(name: str) -> str:
        name = re.sub(r"\.(srt|vtt|ass|ssa|mkv|mp4|avi)$", "", name, flags=re.I)
        name = re.sub(r"\.(zh|en|chs|cht|chi|eng)([-.]?(hans|hant|simp|trad))?", "", name, flags=re.I)
        return name.lower()

    a, b = _clean(subtitle_name), _clean(video_name)
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


class ActiveTrackRegistry:
    """活跃字幕轨注册表。线程安全。"""

    def __init__(self, auto_advance: bool = False) -> None:
        self._auto_advance = auto_advance
        self._lock = threading.Lock()
        self._tracks: dict[str, SubtitleTrack] = {}
        self._active_id: str | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def register(self, track: SubtitleTrack) -> None:
        """注册（或更新）一条字幕轨；首条自动成为活跃轨。"""
        with self._lock:
            is_new = track.track_id not in self._tracks
            self._tracks[track.track_id] = track
            if self._active_id is None and is_new:
                self._active_id = track.track_id
        logger.info(
            "字幕轨注册：%s（%s 行）%s",
            track.track_id, track.line_count,
            "→ 设为活跃轨" if self._active_id == track.track_id else "",
        )

    def unregister(self, track_id: str) -> None:
        """注销；注销活跃轨时清空活跃态（不自动指定继任）。"""
        with self._lock:
            self._tracks.pop(track_id, None)
            if self._active_id == track_id:
                self._active_id = None
        logger.info("字幕轨注销：%s", track_id)

    def set_active(self, track_id: str) -> bool:
        """手动切换活跃轨；轨不存在返回 False。"""
        with self._lock:
            if track_id not in self._tracks:
                return False
            changed = self._active_id != track_id
            self._active_id = track_id
        if changed:
            logger.info("活跃字幕轨切换 → %s", track_id)
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def active_id(self) -> str | None:
        with self._lock:
            return self._active_id

    def active_track(self) -> SubtitleTrack | None:
        with self._lock:
            if self._active_id is None:
                return None
            return self._tracks.get(self._active_id)

    def all_tracks(self) -> list[SubtitleTrack]:
        with self._lock:
            return list(self._tracks.values())

    # ------------------------------------------------------------------
    # 推荐（只建议，不静默切换）
    # ------------------------------------------------------------------

    def suggest_for_video(self, video_name: str) -> SubtitleTrack | None:
        """给定正在播放的视频文件名，推荐最匹配的已注册字幕轨。

        规则：剧集号一致（或双方都无标记）时取文件名相似度最高者，
        相似度低于 0.5 不推荐。**只返回建议，不改变活跃态。**
        """
        best, best_score = None, 0.0
        with self._lock:
            tracks = list(self._tracks.values())
        for track in tracks:
            track_ep, video_ep = _episode_of(track.title), _episode_of(video_name)
            if track_ep is not None and video_ep is not None and track_ep != video_ep:
                continue  # 剧集号不同的直接排除
            score = similar_title_score(track.title, video_name)
            if score > best_score:
                best, best_score = track, score
        if best is None or best_score < 0.5:
            return None
        return best

    def auto_advance(self, current_video: str, sibling_files: list[str]) -> SubtitleTrack | None:
        """连播：当前活跃轨接近轨尾时，找下一集字幕并（按配置）切换。

        严格连续才动作；返回切到的轨（未开启自动连播时返回 None，
        由 UI 提示用户确认）。
        """
        with self._lock:
            active = (
                self._tracks.get(self._active_id)
                if self._active_id is not None else None
            )
        if active is None:
            return None
        next_name = next_episode_file(current_video, sibling_files)
        if next_name is None:
            return None
        candidate = self.suggest_for_video(next_name)
        if candidate is None:
            return None
        if self._auto_advance:
            self.set_active(candidate.track_id)
            return candidate
        return None
