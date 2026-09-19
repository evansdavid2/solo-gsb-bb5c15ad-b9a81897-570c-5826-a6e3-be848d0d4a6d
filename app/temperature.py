"""温控责任审计。

处理流程：
1. 调用现有单链校验器（``validate_events``）得到原校验结果并原样保留；
2. 仅以其中 ``status=ok`` 的 seal / reseal / handover 事件生成保管人半开区间
   ``[start, end)``：每个有效事件以其时间戳开启对应保管人的区间，下一个有效
   事件闭合前一区间，最后一个保管人区间向右开放（延伸至无穷）。unseal 不改变
   保管人，违规事件一律不产生区间；
3. 温度样本时间戳必须严格递增（否则 422）；相邻样本间隔不超过
   ``max_gap_seconds`` 时按前值保持形成半开区间 ``[t_i, t_{i+1})``，超出阈值
   视为数据空洞、不计入任何时长；最后一个样本不形成区间；
4. 起始样本温度高于 ``max_temperature`` 的区间为高温越界（high），低于
   ``min_temperature`` 的为低温越界（low），恰好等于阈值视为在范围内；
   用公共层 :func:`app.timeline.split_interval` 把每段越界区间与保管人
   覆盖区间求交，跨保管人区间边界（交接点）必须拆分为多段，未被任何
   保管人区间覆盖的部分计入未归属；
5. 按保管人汇总高温 / 低温秒数与各自最大偏差，返回逐段明细（含起始样本
   下标）、全局最大偏差与未归属总秒数。

事件与样本的时间戳不得混用带时区与不带时区的时间（否则 422）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .models import (
    CustodianTemperatureSummary,
    Event,
    EventResult,
    EventType,
    TemperatureAuditRequest,
    TemperatureAuditResponse,
    TemperatureBreachSegment,
    TemperatureMaxDeviation,
    TemperatureSample,
)
from .timeline import (
    CoverInterval,
    ensure_comparable,
    split_interval,
    valid_events,
)
from .validator import validate_events


class TemperatureAuditError(ValueError):
    """审计入参的语义错误（在路由层转换为 HTTP 422）。"""


@dataclass(frozen=True)
class _BreachInterval:
    """一段由单个起始样本前值保持形成的越界半开区间 [start, end)。"""

    start: datetime
    end: datetime
    kind: str  # "high" / "low"
    temperature: float
    deviation: float
    sample_index: int


@dataclass(frozen=True)
class _CustodianInfo:
    """保管人覆盖区间的领域归属：保管人及开启区间的有效事件全链下标。"""

    custodian: str
    event_index: int


def _derive_custodian_covers(
    events: List[Event], event_results: List[EventResult]
) -> List[CoverInterval[_CustodianInfo]]:
    """从 status=ok 的 seal/reseal/handover 事件生成保管人覆盖区间。

    事件过滤与全链下标保留由公共层 :func:`app.timeline.valid_events` 完成；
    每个有效事件开启对应保管人的半开覆盖区间，下一个有效事件闭合前一区间
    （有效事件时间戳严格递增，区间必然按起点有序且首尾相接），最后一个
    区间以 ``end=None`` 向右开放。unseal 与违规事件不产生区间。
    """
    points = [
        (selected.event.timestamp, selected.event.custodian, selected.index)
        for selected in valid_events(
            events,
            event_results,
            types=(EventType.SEAL, EventType.RESEAL, EventType.HANDOVER),
        )
    ]

    covers: List[CoverInterval[_CustodianInfo]] = []
    for i, (start, custodian, event_index) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else None
        covers.append(
            CoverInterval(
                start=start,
                end=end,
                payload=_CustodianInfo(
                    custodian=custodian,  # type: ignore[arg-type]
                    event_index=event_index,
                ),
            )
        )
    return covers


def _derive_breach_intervals(
    samples: List[TemperatureSample],
    min_temperature: float,
    max_temperature: float,
    max_gap_seconds: float,
) -> List[_BreachInterval]:
    """按前值保持从相邻样本推导越界区间；数据空洞与在范围内的区间不计。"""
    breaches: List[_BreachInterval] = []
    for i in range(len(samples) - 1):
        current, nxt = samples[i], samples[i + 1]
        gap = (nxt.timestamp - current.timestamp).total_seconds()
        if gap > max_gap_seconds:
            continue  # 数据空洞，不计时
        temperature = current.temperature
        if temperature > max_temperature:
            kind, deviation = "high", temperature - max_temperature
        elif temperature < min_temperature:
            kind, deviation = "low", min_temperature - temperature
        else:
            continue  # 在允许范围内（含恰好等于阈值）
        breaches.append(
            _BreachInterval(
                start=current.timestamp,
                end=nxt.timestamp,
                kind=kind,
                temperature=temperature,
                deviation=deviation,
                sample_index=i,
            )
        )
    return breaches


def _split_by_custodian(
    breach: _BreachInterval, covers: List[CoverInterval[_CustodianInfo]]
) -> List[TemperatureBreachSegment]:
    """把一段越界区间按保管人覆盖区间边界拆分（跨交接点必须拆分）。

    切分由公共层 :func:`app.timeline.split_interval` 统一完成（覆盖区间
    首尾相接、最后一个向右开放）；这里把通用 covered/gap 片段映射为温控
    明细：covered 归属命中保管人，gap 为未归属段。
    """
    pieces = split_interval(breach.start, breach.end, covers)
    return [
        TemperatureBreachSegment(
            start=piece.start,
            end=piece.end,
            seconds=(piece.end - piece.start).total_seconds(),
            kind=breach.kind,
            temperature=breach.temperature,
            deviation=breach.deviation,
            sample_index=breach.sample_index,
            custodian=piece.payload.custodian if piece.payload is not None else None,
            custodian_event_index=(
                piece.payload.event_index if piece.payload is not None else None
            ),
        )
        for piece in pieces
    ]


def _ensure_strictly_increasing(samples: List[TemperatureSample]) -> None:
    """温度样本时间戳必须严格递增，否则 422。"""
    for i in range(1, len(samples)):
        if samples[i].timestamp <= samples[i - 1].timestamp:
            raise TemperatureAuditError(
                "温度样本时间戳必须严格递增："
                f"下标 {i} 的样本 {samples[i].timestamp.isoformat()} 未晚于"
                f"下标 {i - 1} 的样本 {samples[i - 1].timestamp.isoformat()}"
            )


def audit_temperature(request: TemperatureAuditRequest) -> TemperatureAuditResponse:
    """校验事件链并把越界温度时长归属到保管人。"""
    validation = validate_events(request.events)
    timestamps = [event.timestamp for event in request.events]
    timestamps.extend(sample.timestamp for sample in request.samples)
    ensure_comparable(timestamps, TemperatureAuditError)
    _ensure_strictly_increasing(request.samples)

    custodian_covers = _derive_custodian_covers(request.events, validation.events)
    breaches = _derive_breach_intervals(
        request.samples,
        request.min_temperature,
        request.max_temperature,
        request.max_gap_seconds,
    )

    segments: List[TemperatureBreachSegment] = []
    for breach in breaches:  # 越界区间按样本顺序产生，本身即时间升序
        segments.extend(_split_by_custodian(breach, custodian_covers))

    # 按保管人汇总（按首次归属时间升序，仅包含有越界时长的保管人）
    summaries: Dict[str, CustodianTemperatureSummary] = {}
    unattributed_seconds = 0.0
    for segment in segments:
        if segment.custodian is None:
            unattributed_seconds += segment.seconds
            continue
        summary = summaries.get(segment.custodian)
        if summary is None:
            summary = summaries[segment.custodian] = CustodianTemperatureSummary(
                custodian=segment.custodian,
                high_seconds=0.0,
                low_seconds=0.0,
                max_high_deviation=None,
                max_low_deviation=None,
            )
        if segment.kind == "high":
            summary.high_seconds += segment.seconds
            if summary.max_high_deviation is None or segment.deviation > summary.max_high_deviation:
                summary.max_high_deviation = segment.deviation
        else:
            summary.low_seconds += segment.seconds
            if summary.max_low_deviation is None or segment.deviation > summary.max_low_deviation:
                summary.max_low_deviation = segment.deviation

    # 全局最大偏差（含未归属段；并列时取时间最早者）
    max_deviation: Optional[TemperatureMaxDeviation] = None
    for breach in breaches:
        if max_deviation is None or breach.deviation > max_deviation.deviation:
            max_deviation = TemperatureMaxDeviation(
                kind=breach.kind,
                deviation=breach.deviation,
                sample_index=breach.sample_index,
                timestamp=breach.start,
            )

    return TemperatureAuditResponse(
        validation=validation,
        segments=segments,
        custodians=list(summaries.values()),
        unattributed_seconds=unattributed_seconds,
        max_deviation=max_deviation,
    )
