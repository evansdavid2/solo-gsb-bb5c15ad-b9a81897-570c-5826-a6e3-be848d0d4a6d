"""公共时间线原语（app/timeline.py）的单元测试，以及三个审计端点的
参数化回归测试：违规事件跳过、嵌套 / 相接窗口、交接点切分、未归属段与
混合时区。"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.exposure import ExposureAuditError, audit_exposure
from app.handover import HandoverPlanAuditError, audit_handover_plan
from app.models import (
    AllowedWindow,
    Event,
    EventResult,
    EventType,
    ExposureAuditRequest,
    HandoverPlanAuditRequest,
    HandoverPlanNode,
    TemperatureAuditRequest,
    TemperatureSample,
)
from app.temperature import TemperatureAuditError, audit_temperature
from app.timeline import (
    CoverInterval,
    TimelinePiece,
    ensure_comparable,
    split_interval,
    valid_events,
)

# ---------------------------------------------------------------------------
# 测试时间线原语：统一使用 naive 时间（领域无关，不涉及时区语义）
# ---------------------------------------------------------------------------

BASE = datetime(2026, 9, 19, 0, 0, 0)


def t(hour: int, minute: int = 0) -> datetime:
    return BASE.replace(hour=hour, minute=minute)


def cover(start_h: int, end_h, payload=None) -> CoverInterval:
    return CoverInterval(start=t(start_h), end=None if end_h is None else t(end_h),
                         payload=payload)


def piece_tuples(pieces):
    return [
        (
            p.start.strftime("%H:%M"),
            p.end.strftime("%H:%M"),
            p.covered,
            p.payload,
        )
        for p in pieces
    ]


# ---------------------------------------------------------------------------
# split_interval：covered / gap 切分、相接边界、开放末端
# ---------------------------------------------------------------------------


def test_split_single_cover_with_gaps_on_both_sides():
    pieces = split_interval(t(10), t(20), [cover(12, 14, "W")])
    assert piece_tuples(pieces) == [
        ("10:00", "12:00", False, None),
        ("12:00", "14:00", True, "W"),
        ("14:00", "20:00", False, None),
    ]


def test_split_multiple_covers_with_gaps():
    pieces = split_interval(
        t(10), t(20), [cover(11, 12, "A"), cover(14, 16, "B")]
    )
    assert piece_tuples(pieces) == [
        ("10:00", "11:00", False, None),
        ("11:00", "12:00", True, "A"),
        ("12:00", "14:00", False, None),
        ("14:00", "16:00", True, "B"),
        ("16:00", "20:00", False, None),
    ]


def test_split_abutting_covers_produce_no_gap_piece():
    # [11,13) 与 [13,15) 在 13:00 相接：半开语义下不产生零长度 gap
    pieces = split_interval(t(10), t(16), [cover(11, 13, "A"), cover(13, 15, "B")])
    assert piece_tuples(pieces) == [
        ("10:00", "11:00", False, None),
        ("11:00", "13:00", True, "A"),
        ("13:00", "15:00", True, "B"),
        ("15:00", "16:00", False, None),
    ]


def test_split_nested_cover_is_skipped_after_parent_consumes_cursor():
    # [11,16) 包住 [12,13)：内层区间在游标 16:00 之前，整体跳过，不重复切段
    pieces = split_interval(t(10), t(18), [cover(11, 16, "outer"), cover(12, 13, "inner")])
    assert piece_tuples(pieces) == [
        ("10:00", "11:00", False, None),
        ("11:00", "16:00", True, "outer"),
        ("16:00", "18:00", False, None),
    ]


def test_split_open_ended_cover_extends_to_target_end():
    pieces = split_interval(t(10), t(20), [cover(12, None, "open")])
    assert piece_tuples(pieces) == [
        ("10:00", "12:00", False, None),
        ("12:00", "20:00", True, "open"),
    ]


def test_split_open_ended_last_cover_after_gap():
    # [10,12) 之后是向右开放的区间：gap 后 covered 延伸到目标末端
    pieces = split_interval(
        t(10), t(20), [cover(10, 12, "A"), cover(15, None, "B")]
    )
    assert piece_tuples(pieces) == [
        ("10:00", "12:00", True, "A"),
        ("12:00", "15:00", False, None),
        ("15:00", "20:00", True, "B"),
    ]


def test_split_covers_outside_target_are_ignored():
    # 覆盖区间整体落在目标之后：整个目标为 gap
    pieces = split_interval(t(10), t(11), [cover(12, 13, "late")])
    assert piece_tuples(pieces) == [("10:00", "11:00", False, None)]
    # 覆盖区间整体在游标之前（end <= cursor）：同样不切段
    pieces = split_interval(t(10), t(20), [cover(8, 9, "early")])
    assert piece_tuples(pieces) == [("10:00", "20:00", False, None)]
    # 覆盖区间起点恰等于目标末端（半开）：不覆盖任何时长
    pieces = split_interval(t(10), t(12), [cover(12, 13, "touch")])
    assert piece_tuples(pieces) == [("10:00", "12:00", False, None)]


def test_split_cover_is_clipped_to_target_bounds():
    pieces = split_interval(t(11), t(13), [cover(0, 23, "W")])
    assert piece_tuples(pieces) == [("11:00", "13:00", True, "W")]
    # 开放覆盖区间也按目标末端裁剪
    pieces = split_interval(t(11), t(13), [cover(0, None, "W")])
    assert piece_tuples(pieces) == [("11:00", "13:00", True, "W")]


def test_split_empty_target_returns_no_pieces():
    assert split_interval(t(12), t(12), [cover(0, 23, "W")]) == []
    assert split_interval(t(12), t(11), [cover(0, None, "W")]) == []


def test_split_no_covers_makes_single_gap():
    pieces = split_interval(t(10), t(20), [])
    assert piece_tuples(pieces) == [("10:00", "20:00", False, None)]


def test_split_pieces_tile_the_target_without_overlap():
    covers = [cover(11, 12, "A"), cover(14, None, "B")]
    pieces = split_interval(t(10), t(20), covers)
    # 相邻片段首尾相接，整体恰好铺满目标区间
    assert pieces[0].start == t(10)
    assert pieces[-1].end == t(20)
    for prev, nxt in zip(pieces, pieces[1:]):
        assert prev.end == nxt.start


# ---------------------------------------------------------------------------
# valid_events：按 status=ok 过滤并保留全链下标
# ---------------------------------------------------------------------------


def _event(type_: EventType, hour: int) -> Event:
    return Event(type=type_, timestamp=t(hour))


def _result(index: int, type_: EventType, status: str) -> EventResult:
    return EventResult(index=index, type=type_, status=status)


def test_valid_events_skips_violations_and_keeps_chain_indices():
    events = [
        _event(EventType.UNSEAL, 9),     # 违规
        _event(EventType.SEAL, 10),      # ok
        _event(EventType.HANDOVER, 11),  # ok
        _event(EventType.SEAL, 12),      # 违规
    ]
    results = [
        _result(0, EventType.UNSEAL, "violation"),
        _result(1, EventType.SEAL, "ok"),
        _result(2, EventType.HANDOVER, "ok"),
        _result(3, EventType.SEAL, "violation"),
    ]
    selected = valid_events(events, results)
    assert [(v.index, v.event.type) for v in selected] == [
        (1, EventType.SEAL),
        (2, EventType.HANDOVER),
    ]


def test_valid_events_type_filter():
    events = [
        _event(EventType.SEAL, 10),
        _event(EventType.UNSEAL, 11),
        _event(EventType.RESEAL, 12),
        _event(EventType.HANDOVER, 13),
    ]
    results = [_result(i, e.type, "ok") for i, e in enumerate(events)]
    selected = valid_events(
        events, results, types=(EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)
    )
    assert [v.index for v in selected] == [0, 2, 3]
    # unseal/reseal 过滤
    selected = valid_events(
        events, results, types=(EventType.UNSEAL, EventType.RESEAL)
    )
    assert [v.index for v in selected] == [1, 2]


def test_valid_events_empty():
    assert valid_events([], []) == []


# ---------------------------------------------------------------------------
# ensure_comparable：统一时区可比性检查
# ---------------------------------------------------------------------------


def test_ensure_comparable_all_aware_or_all_naive_passes():
    ensure_comparable([t(10)], RuntimeError)  # 单个值
    ensure_comparable(
        [
            datetime(2026, 9, 19, 10, tzinfo=datetime.now().astimezone().tzinfo),
            datetime(2026, 9, 19, 11, tzinfo=datetime.now().astimezone().tzinfo),
        ],
        RuntimeError,
    )
    ensure_comparable([], RuntimeError)


def test_ensure_comparable_mixed_raises_callers_error_type_and_message():
    class AuditError(ValueError):
        pass

    values = [
        datetime(2026, 9, 19, 10),  # naive
        datetime(2026, 9, 19, 11).astimezone(),  # aware
    ]
    with pytest.raises(AuditError) as excinfo:
        ensure_comparable(values, AuditError)
    assert "时区" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 三个审计端点的参数化回归：公共链（含交错的违规事件）
# ---------------------------------------------------------------------------

AWARE_DAY = "2026-09-19"


def adt(hour: int, minute: int = 0, aware: bool = True) -> datetime:
    suffix = "+00:00" if aware else ""
    return datetime.fromisoformat(f"{AWARE_DAY}T{hour:02d}:{minute:02d}:00{suffix}")


def aev(type_: EventType, hour: int, minute: int = 0, aware: bool = True, **kw) -> Event:
    return Event(type=type_, timestamp=adt(hour, minute, aware), **kw)


def shared_chain():
    """跨三个端点共用的事件链，违规事件穿插在有效事件之间：

    0 seal@08 Alice（ok，首保管人区间起点）
    1 seal@08:30 X（violation：已封装重复封装，跳过）
    2 unseal@09（ok，开启唯一开封区间）
    3 handover@09:30 Bob（violation：未封装交接，跳过）
    4 reseal@10 Alice（ok，闭合开封区间；新保管人区间起点）
    5 handover@11 Bob（ok，交接点）
    """
    return [
        aev(EventType.SEAL, 8, seal_id="S001", custodian="Alice"),      # 0 ok
        aev(EventType.SEAL, 8, 30, seal_id="S999", custodian="X"),      # 1 violation（重复封装）
        aev(EventType.UNSEAL, 9),                                       # 2 ok
        aev(EventType.HANDOVER, 9, 30, custodian="Bob"),                # 3 violation（未封装交接）
        aev(EventType.RESEAL, 10, seal_id="S002", custodian="Alice"),  # 4 ok
        aev(EventType.HANDOVER, 11, custodian="Bob"),                   # 5 ok
    ]


VIOLATION_INDICES = [1, 3]


def exposure_segments(resp):
    return [
        (s.start.strftime("%H:%M"), s.end.strftime("%H:%M"),
         s.unseal_index, s.reseal_index)
        for s in resp.covered + resp.uncovered
    ]


def temperature_segments(resp):
    return [
        (s.start.strftime("%H:%M"), s.end.strftime("%H:%M"), s.custodian,
         s.custodian_event_index, s.sample_index)
        for s in resp.segments
    ]


def test_shared_chain_validation_is_preserved_identically():
    """三个端点保留的 validation 完全一致，且违规下标为 [1, 3]。"""
    chain = shared_chain()

    expo = audit_exposure(
        ExposureAuditRequest(
            events=chain, audit_until=adt(20), allowed_windows=[]
        )
    )
    hand = audit_handover_plan(HandoverPlanAuditRequest(events=chain, plan=[]))
    temp = audit_temperature(
        TemperatureAuditRequest(
            events=chain, samples=[], min_temperature=2.0,
            max_temperature=8.0, max_gap_seconds=600,
        )
    )

    for resp in (expo.validation, hand.validation, temp.validation):
        assert resp.valid is False
        assert [v.index for v in resp.violations] == VIOLATION_INDICES
        assert [r.index for r in resp.events if r.status == "ok"] == [0, 2, 4, 5]


@pytest.mark.parametrize(
    "windows",
    [
        # 相接窗口：[08:30,09:30) 与 [09:30,11:00) 合并为 [08:30,11:00)
        [
            AllowedWindow(start=adt(8, 30), end=adt(9, 30)),
            AllowedWindow(start=adt(9, 30), end=adt(11)),
        ],
        # 嵌套窗口（乱序给出）：[09,11) 包住 [09:30,10:30)
        [
            AllowedWindow(start=adt(9, 30), end=adt(10, 30)),
            AllowedWindow(start=adt(9), end=adt(11)),
        ],
    ],
)
def test_exposure_nested_and_abutting_windows_regression(windows):
    """违规事件跳过后唯一开封区间为 [09:00,10:00)（unseal=2, reseal=4）；
    相接与嵌套窗口合并后都完整覆盖该区间。"""
    resp = audit_exposure(
        ExposureAuditRequest(
            events=shared_chain(), audit_until=adt(20), allowed_windows=windows
        )
    )
    assert [(s.start.strftime("%H:%M"), s.end.strftime("%H:%M"),
             s.unseal_index, s.reseal_index) for s in resp.covered] == [
        ("09:00", "10:00", 2, 4)
    ]
    assert resp.uncovered == []
    assert resp.unauthorized_seconds == 0.0


def test_exposure_violation_events_skipped_in_open_interval_indices():
    """违规的 handover（下标 3）位于 unseal 与 reseal 之间，不得改变配对下标。"""
    chain = shared_chain()
    resp = audit_exposure(
        ExposureAuditRequest(events=chain, audit_until=adt(20), allowed_windows=[])
    )
    assert exposure_segments(resp) == [("09:00", "10:00", 2, 4)]
    assert resp.unauthorized_seconds == 3600.0


def test_temperature_handover_point_split_and_unattributed_regression():
    """越界区间 [10:30,11:30) 跨交接点 11:00（事件下标 5）切分；
    前段 [10:30,11:00) 归 Alice（事件 4 开启的区间），后段归 Bob（事件 5）。
    另取一段 [07:30,08:30)：首保管人区间 08:00 之前为未归属段。"""
    samples = [
        TemperatureSample(timestamp=adt(7, 30), temperature=10.0),
        TemperatureSample(timestamp=adt(8, 30), temperature=5.0),
        TemperatureSample(timestamp=adt(10, 30), temperature=10.0),
        TemperatureSample(timestamp=adt(11, 30), temperature=5.0),
    ]
    resp = audit_temperature(
        TemperatureAuditRequest(
            events=shared_chain(), samples=samples,
            min_temperature=2.0, max_temperature=8.0, max_gap_seconds=7200,
        )
    )
    # 注意 [08:30,10:30) 间隔恰好 7200 不超阈值，但温度 5.0 在范围内，不形成越界
    assert temperature_segments(resp) == [
        ("07:30", "08:00", None, None, 0),       # 未归属
        ("08:00", "08:30", "Alice", 0, 0),       # 归首个有效事件
        ("10:30", "11:00", "Alice", 4, 2),       # 交接点前
        ("11:00", "11:30", "Bob", 5, 2),         # 交接点后
    ]
    assert resp.unattributed_seconds == 1800.0
    summaries = {s.custodian: s for s in resp.custodians}
    assert summaries["Alice"].high_seconds == 3600.0
    assert summaries["Bob"].high_seconds == 1800.0


def test_temperature_open_ended_last_custodian_covers_far_samples():
    """末个保管人区间向右开放：远晚于最后事件的越界样本归 Bob（事件下标 5）。"""
    samples = [
        TemperatureSample(timestamp=adt(20), temperature=10.0),
        TemperatureSample(timestamp=adt(20, 10), temperature=5.0),
    ]
    resp = audit_temperature(
        TemperatureAuditRequest(
            events=shared_chain(), samples=samples,
            min_temperature=2.0, max_temperature=8.0, max_gap_seconds=600,
        )
    )
    assert temperature_segments(resp) == [("20:00", "20:10", "Bob", 5, 0)]
    assert resp.unattributed_seconds == 0.0


def test_handover_violation_events_skipped_in_arrivals():
    """到任记录只取 ok 的 seal/reseal/handover（下标 0/4/5），
    违规的下标 1/3 与 unseal 下标 2 均不参与匹配。"""
    plan = [
        HandoverPlanNode(custodian="Alice",
                         window_start=adt(8), window_end=adt(9)),
        HandoverPlanNode(custodian="Alice",
                         window_start=adt(10), window_end=adt(11)),
        HandoverPlanNode(custodian="Bob",
                         window_start=adt(11), window_end=adt(12)),
    ]
    resp = audit_handover_plan(
        HandoverPlanAuditRequest(events=shared_chain(), plan=plan)
    )
    results = {r.index: r for r in resp.plan_results}
    assert results[0].status == "matched" and results[0].event_index == 0
    assert results[1].status == "matched" and results[1].event_index == 4
    assert results[2].status == "matched" and results[2].event_index == 5
    assert resp.summary.matched == 3
    assert resp.unexpected == []


@pytest.mark.parametrize(
    "endpoint,request_factory,error_type",
    [
        (
            "exposure",
            lambda: ExposureAuditRequest(
                events=[aev(EventType.SEAL, 8, aware=True,
                            seal_id="S1", custodian="Alice")],
                audit_until=adt(20, aware=False),  # naive 与 aware 事件混用
                allowed_windows=[],
            ),
            ExposureAuditError,
        ),
        (
            "temperature",
            lambda: TemperatureAuditRequest(
                events=[aev(EventType.SEAL, 8, aware=True,
                            seal_id="S1", custodian="Alice")],
                samples=[TemperatureSample(timestamp=adt(9, aware=False),
                                           temperature=10.0)],
                min_temperature=2.0,
                max_temperature=8.0,
                max_gap_seconds=600,
            ),
            TemperatureAuditError,
        ),
        (
            "handover",
            lambda: HandoverPlanAuditRequest.model_construct(
                events=[aev(EventType.SEAL, 8, aware=True,
                            seal_id="S1", custodian="Alice")],
                # 窗口自身起止混用 naive/aware 会先被模型校验拦截，
                # 用 model_construct 模拟 JSON 直入以触达审计层的统一时区检查
                plan=[HandoverPlanNode.model_construct(
                    custodian="Alice",
                    window_start=adt(10, aware=True),
                    window_end=adt(11, aware=False),
                )],
            ),
            HandoverPlanAuditError,
        ),
    ],
)
def test_mixed_timezones_raise_domain_error_with_shared_message(
    endpoint, request_factory, error_type
):
    """三个端点对混合时区统一经 timeline.ensure_comparable 报各自的领域错误，
    422 detail 文案保持不变。"""
    request = request_factory()
    with pytest.raises(error_type) as excinfo:
        if endpoint == "exposure":
            audit_exposure(request)
        elif endpoint == "temperature":
            audit_temperature(request)
        else:
            audit_handover_plan(request)
    assert "时间戳时区信息不一致" in str(excinfo.value)
