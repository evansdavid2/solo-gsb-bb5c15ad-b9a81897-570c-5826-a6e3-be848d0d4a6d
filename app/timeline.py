"""审计端点共用的时间线原语（领域无关）。

开封暴露 / 交接计划 / 温控责任三个审计端点共用三类逻辑：

1. 按单链校验结果过滤 ``status=ok`` 事件，并保留事件在**全链事件**中的
   下标（:func:`valid_events`），违规事件一律跳过；
2. 半开区间分割器（:func:`split_interval`）：把一个目标半开区间
   ``[start, end)`` 按一组按起点有序的覆盖区间切成 covered / gap 片段——
   覆盖区间允许相交前先由调用方合并、允许在边界处相接（相接不产生零长度
   gap），并支持以 ``end=None`` 表示向右开放的末端；
3. 时区可比性检查（:func:`ensure_comparable`）：一组 ``datetime`` 不得
   混用带时区与不带时区的时间，错误类型由调用方给定。

本模块只返回领域无关记录（事件下标 + 事件、通用半开片段），不引用任何
审计响应模型；暴露、到任、温控等领域字段由调用方自行映射。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Container, Generic, List, Optional, Sequence, TypeVar

from .models import Event, EventResult

T = TypeVar("T")
P = TypeVar("P")

#: 三个审计端点共用的混用时区报错文案（与历史 422 detail 保持一致）。
MIXED_TIMEZONE_MESSAGE = "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"


@dataclass(frozen=True)
class ValidEvent(Generic[T]):
    """一条 ``status=ok`` 事件及其在全链事件中的下标（从 0 开始）。"""

    index: int
    event: T


@dataclass(frozen=True)
class CoverInterval(Generic[P]):
    """一段半开覆盖区间 ``[start, end)``；``end`` 为 None 表示向右开放。

    调用方通过 ``payload`` 携带该区间的领域归属（如授权窗口或保管人），
    分割后的 covered 片段会原样带回该归属；gap 片段的归属为 None。
    """

    start: datetime
    end: Optional[datetime]
    payload: P


@dataclass(frozen=True)
class TimelinePiece(Generic[P]):
    """目标区间被切出的一个半开片段 ``[start, end)``。

    ``covered`` 为 True 表示该片段落在某个覆盖区间内（``payload`` 为该
    覆盖区间的归属）；False 表示未被任何覆盖区间覆盖的间隙（gap）。
    """

    start: datetime
    end: datetime
    covered: bool
    payload: Optional[P]


def valid_events(
    events: Sequence[Event],
    event_results: Sequence[EventResult],
    types: Optional[Container] = None,
) -> List[ValidEvent[Event]]:
    """按校验结果过滤 ``status=ok`` 事件，保留全链下标。

    违规事件一律跳过；给定 ``types`` 时仅保留这些事件类型。事件与逐事件
    结果按下标一一对应（与 ``zip`` 相同），返回顺序即事件输入顺序。
    """
    selected: List[ValidEvent[Event]] = []
    for event, result in zip(events, event_results):
        if result.status != "ok":
            continue
        if types is not None and event.type not in types:
            continue
        selected.append(ValidEvent(index=result.index, event=event))
    return selected


def split_interval(
    start: datetime,
    end: datetime,
    covers: Sequence[CoverInterval[P]],
) -> List[TimelinePiece[P]]:
    """把目标半开区间 ``[start, end)`` 按有序覆盖区间切成 covered/gap 片段。

    覆盖区间须按起点升序；彼此可相接（``a.end == b.start``，相接点两侧均
    被覆盖，不会产生零长度 gap），也可留有间隙（间隙成为 gap 片段）。
    ``end=None`` 的覆盖区间视为向右开放（末端不超过目标区间终点）。
    零长度（``start >= end``）的目标区间不产生任何片段。

    返回片段恰好拼接成原目标区间、互不重叠且按时间升序；
    每个 covered 片段携带命中覆盖区间的 ``payload``，gap 片段为 None。
    """
    pieces: List[TimelinePiece[P]] = []
    if start >= end:
        return pieces

    cursor = start
    for cover in covers:
        if cursor >= end:
            break
        cover_end = end if cover.end is None else cover.end
        if cover_end <= cursor:
            continue  # 覆盖区间整体在游标之前
        if cover.start >= end:
            break  # 覆盖区间整体在目标区间之后

        if cover.start > cursor:
            # 覆盖区间开始前的部分未被覆盖（gap）；start == cursor 即相接，不切段
            pieces.append(TimelinePiece(cursor, cover.start, False, None))
        overlap_start = max(cursor, cover.start)
        overlap_end = min(end, cover_end)
        if overlap_start < overlap_end:
            pieces.append(TimelinePiece(overlap_start, overlap_end, True, cover.payload))
            cursor = overlap_end

    if cursor < end:
        pieces.append(TimelinePiece(cursor, end, False, None))
    return pieces


def ensure_comparable(
    values: Sequence[datetime], error_factory: Callable[[str], Exception]
) -> None:
    """检查一组时间戳的时区信息一致（不得混用带时区与不带时区的时间）。

    不一致时调用 ``error_factory``（由调用方传入其领域错误类型）抛出，
    以便在路由层统一转换为 HTTP 422。
    """
    if not values:
        return
    aware = _is_timezone_aware(values[0])
    for value in values[1:]:
        if _is_timezone_aware(value) != aware:
            raise error_factory(MIXED_TIMEZONE_MESSAGE)


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None
