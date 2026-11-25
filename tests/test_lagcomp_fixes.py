"""六个缺陷的回归测试（与修复点 1~6 对应）。"""

import copy

import pytest

from gameplay.lagcomp_rewind_hit import (
    DEFAULT_HISTORY_MS,
    Entity,
    PeerState,
    Registry,
    SnapshotStore,
    clamp_interp,
    confirm,
    determinism,
    drop_stale_seq,
    fallback,
    isolate_request,
    metrics,
    record_snapshot,
    release,
    resolve_hit,
    rewind,
    snapshot_by_ts,
    validate_hit,
    validate_handle,
)


def make_store(**kwargs):
    return SnapshotStore(history_ms=1000, max_rewind_ms=250, **kwargs)


def make_registry():
    reg = Registry()
    reg.add(Entity(eid=1, pos=(0.0, 0.0), radius=1.0))
    reg.add(Entity(eid=2, pos=(10.0, 0.0), radius=1.0))
    return reg


def feed(store, registry, timestamps, mover_eid=2, step=1.0):
    """在每个时间戳记录快照，eid=2 沿 +x 匀速移动。"""
    for i, ts in enumerate(timestamps):
        reg_entity = registry.get(mover_eid)
        reg_entity.pos = (10.0 + i * step, 0.0)
        record_snapshot(store, ts, registry.entities)


# --- 修复点 1：快照按时间戳索引，不被帧号环覆盖 ---------------------------

def test_snapshot_indexed_by_timestamp_not_overwritten():
    store, reg = make_store(), make_registry()
    feed(store, reg, [0, 16, 32, 48, 64])

    # 所有历史时间戳都能按时间戳取回，不存在"最旧帧被覆盖"
    for ts in (0, 16, 32, 48, 64):
        hit = snapshot_by_ts(store, ts)
        assert "exact" in hit and hit["exact"]["ts"] == ts

    # 客户端时钟快半拍：请求时间戳晚于最新帧，钳到最新帧而非乱指旧帧
    future = snapshot_by_ts(store, 80)
    assert future["clamped"] is True
    assert future["before"]["ts"] == 64 and future["after"] is None

    # 超窗口的最旧快照按时间淘汰（不是按帧号被覆盖）
    record_snapshot(store, 0 + DEFAULT_HISTORY_MS + 50, reg.entities)
    assert snapshot_by_ts(store, 0) == {}
    assert store.oldest_ts() > 0


# --- 修复点 2：插值钳制，急停不外推 ---------------------------------------

def test_interpolation_clamped_when_target_suddenly_stops():
    # t=0 在 x=10，t=100 在 x=20，之后急停。请求时间戳落在区间外。
    snap0 = {"ts": 0, "entities": {1: {"pos": (10.0, 0.0), "radius": 1.0,
                                       "alive": True, "gen": 0}}}
    snap1 = {"ts": 100, "entities": {1: {"pos": (20.0, 0.0), "radius": 1.0,
                                         "alive": True, "gen": 0}}}

    # 越界请求：原始 alpha = 1.5 / -0.25，都必须被钳到 1.0 / 0.0
    late = clamp_interp(snap0, snap1, 150)
    assert late["alpha"] == 1.0 and late["clamped"] is True
    assert late["entities"][1]["pos"] == (20.0, 0.0)

    early = clamp_interp(snap0, snap1, -25)
    assert early["alpha"] == 0.0 and early["clamped"] is True
    assert early["entities"][1]["pos"] == (10.0, 0.0)

    # 区间内正常混合不被标记为钳制，且位置严格落在两帧之间
    mid = clamp_interp(snap0, snap1, 50)
    assert mid["alpha"] == 0.5 and mid["clamped"] is False
    x = mid["entities"][1]["pos"][0]
    assert 10.0 <= x <= 20.0

    # 急停场景：snap1 与 snap0 同位置，任何 alpha 都不会外推
    stopped = {"ts": 100, "entities": {1: {"pos": (20.0, 0.0), "radius": 1.0,
                                           "alive": True, "gen": 0}}}
    out = clamp_interp(snap1, stopped, 300)
    assert out["clamped"] is True
    assert out["entities"][1]["pos"] == (20.0, 0.0)


# --- 修复点 3：每请求独立还原，同帧多开火互不污染 --------------------------

def test_same_frame_requests_are_isolated():
    store, reg = make_store(), make_registry()
    feed(store, reg, [0, 16, 32])

    packet_a = {"player_id": 10, "fire_ts": 24, "origin": (0.0, 0.0),
                "dir": (1.0, 0.0), "shooter_eid": 10}
    packet_b = copy.deepcopy(packet_a)

    ctx_a = isolate_request(store, packet_a)
    ctx_b = isolate_request(store, packet_b)

    # 两个工作区是独立对象
    assert ctx_a["world"] is not ctx_b["world"]
    ctx_a["world"]["entities"][2]["pos"] = (-999.0, -999.0)
    ctx_a["scratch"]["polluted"] = True

    # B 完全看不到 A 的改写；共享存储也未被触碰
    assert ctx_b["world"]["entities"][2]["pos"] != (-999.0, -999.0)
    assert "polluted" not in ctx_b["scratch"]
    pristine = snapshot_by_ts(store, 32)["exact"]["entities"][2]["pos"]
    assert pristine == reg.get(2).pos


def test_determinism_stable_ordering_and_purity():
    requests = [
        {"player_id": 3, "seq": 2, "fire_ts": 100},
        {"player_id": 1, "seq": 1, "fire_ts": 100},
        {"player_id": 2, "seq": 0, "fire_ts": 50},
    ]
    ordered = determinism(requests)
    assert [r["player_id"] for r in ordered] == [2, 1, 3]
    # 入参不被修改、不被复用
    assert requests[0]["player_id"] == 3
    assert ordered is not requests


# --- 修复点 4：确认包校验代际句柄 -----------------------------------------

def test_async_confirm_rejects_freed_entity_handle():
    reg = make_registry()
    target = reg.get(2)
    stale_handle = target.handle()          # (2, 0)

    # 判定完成、确认异步返回前，目标死亡并被回收（代际 0 -> 1）
    release(reg, 2)
    assert validate_handle(reg, stale_handle) is False
    assert validate_handle(reg, reg.get(2).handle()) is False  # 已死亡也拒绝

    hit = {"hit_id": "h1", "target_handle": stale_handle, "damage": 50.0}
    outcome = confirm(reg, hit)
    assert outcome["status"] == "rejected_stale_handle"
    assert reg.counters["invalid_handle"] >= 1
    assert reg.get(2).hp == 100.0            # 未结算任何伤害


def test_confirm_accepts_live_handle_and_dedupes():
    reg = make_registry()
    hit = {"hit_id": "h9", "target_handle": (2, 0), "damage": 40.0}
    first = confirm(reg, hit)
    assert first["status"] == "confirmed" and first["remaining_hp"] == 60.0

    again = confirm(reg, dict(hit))   # 确认包重传，凭 hit_id 幂等去重
    assert again["status"] == "duplicate"
    assert reg.counters["confirmed"] == 1 and reg.counters["duplicate_confirm"] == 1


def test_lethal_confirm_releases_and_invalidates_handle():
    reg = make_registry()
    outcome = confirm(reg, {"hit_id": "hK", "target_handle": (2, 0),
                            "damage": 100.0})
    assert outcome["status"] == "confirmed_kill"
    assert reg.get(2).alive is False and reg.get(2).gen == 1
    assert validate_handle(reg, (2, 0)) is False


# --- 修复点 5：旧包按序号丢弃（重传不改写历史） ----------------------------

def test_retransmitted_old_packet_dropped_by_seq():
    peer = PeerState()
    p1 = {"seq": 10, "fire_ts": 100}
    p2 = {"seq": 11, "fire_ts": 116}
    retransmit = {"seq": 10, "fire_ts": 100}   # 丢包重传，带原时间戳

    assert drop_stale_seq(peer, p1)["status"] == "accepted"
    assert drop_stale_seq(peer, p2)["status"] == "accepted"

    gate = drop_stale_seq(peer, retransmit)
    assert gate["status"] == "dropped" and gate["reason"] == "stale_seq"
    assert peer.last_seq == 11
    assert peer.counters["stale_dropped"] == 1


def test_validate_hit_pipeline_drops_stale_before_rewind():
    store, reg, peer = make_store(), make_registry(), PeerState()
    feed(store, reg, [0, 16, 32])
    packet = {"seq": 5, "fire_ts": 16, "origin": (0.0, 0.0),
              "dir": (1.0, 0.0), "shooter_eid": 1, "hit_id": "x"}
    assert validate_hit(store, reg, peer, packet, now=32)["status"] == "hit"

    replay = dict(packet)                       # 同 seq 的重传包
    out = validate_hit(store, reg, peer, replay, now=32)
    assert out["status"] == "dropped"
    # 重传没有产生新的回溯
    assert store.counters["rewind_ok"] == 1


# --- 修复点 6：回溯深度上限 + fallback -------------------------------------

def test_rewind_depth_limit_rejects_ancient_request():
    store, reg = make_store(), make_registry()
    feed(store, reg, [0, 16, 32, 64, 128, 256])

    ok = rewind(store, fire_ts=20, now=256)     # 深度 236ms <= 250ms
    assert ok["status"] == "ok"

    too_deep = rewind(store, fire_ts=0, now=300)  # 深度 300ms > 250ms
    assert too_deep["status"] == "too_deep"
    assert too_deep["depth_ms"] == 300
    assert store.counters["rewind_too_deep"] >= 1
    assert "world" not in too_deep              # 没有偷偷用最旧帧

    fb = fallback(store, {"fire_ts": 0}, reason="rewind_too_deep")
    assert fb["status"] == "fallback_rejected"
    assert store.counters["fallback_rejected"] == 1


def test_fallback_oldest_policy_is_marked_degraded():
    store = make_store(fallback_policy="oldest")
    reg = make_registry()
    feed(store, reg, [0, 16, 32])
    fb = fallback(store, {"fire_ts": -1000}, reason="rewind_too_deep")
    assert fb["status"] == "fallback_oldest" and fb["degraded"] is True
    assert fb["ctx"]["world"]["ts"] == 0


# --- 端到端：正常命中与指标 ------------------------------------------------

def test_end_to_end_hit_and_metrics():
    store, reg, peer = make_store(), make_registry(), PeerState()
    feed(store, reg, [0, 16, 32])

    packet = {
        "seq": 1, "fire_ts": 24, "origin": (0.0, 0.0),
        "dir": (1.0, 0.0), "shooter_eid": 1, "hit_id": "e2e",
    }
    envelope = validate_hit(store, reg, peer, packet, now=32)
    assert envelope["status"] == "hit"
    assert envelope["hit"]["target_eid"] == 2
    # 开火时刻 eid=2 位于 x=10 + (24/16)*1 ≈ 11.5，射线沿 +x 必然相交
    assert envelope["hit"]["point"][0] == pytest.approx(10.5, abs=1e-6)

    # 命中确认时实体仍存活 -> 结算；随后死亡 -> 旧句柄失效
    outcome = confirm(reg, envelope["hit"])
    assert outcome["status"] == "confirmed_kill"
    assert validate_handle(reg, envelope["hit"]["target_handle"]) is False

    m = metrics([store, reg, peer])
    assert m["rewind_ok"] == 1 and m["accepted"] == 1 and m["killed"] == 1


def test_contract_smoke_with_bare_int():
    # 复刻仓库自带契约测试：所有入口用单个 int 冒烟调用必须返回非 None
    import gameplay.lagcomp_rewind_hit as mod
    for name in (
        "snapshot_by_ts", "clamp_interp", "isolate_request", "validate_handle",
        "drop_stale_seq", "record_snapshot", "rewind", "resolve_hit",
        "confirm", "release", "validate_hit", "metrics", "audit",
        "fallback", "determinism",
    ):
        assert getattr(mod, name)(1) is not None
