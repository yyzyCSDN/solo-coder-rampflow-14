"""Snow-clearance fleet planning: priority-ordered sweep scheduling, dump-site
capacity, committed-movement avoidance, emergency preemption of unconfirmed
plans, and immutable execution history."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from json import dumps
from typing import Iterable, Mapping

from .models import Decision
from .timeutil import UTC, Window, utc


class SweepState(str, Enum):
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


LOCKED_STATES = frozenset({SweepState.CONFIRMED, SweepState.EXECUTING, SweepState.COMPLETE})


class SnowPlanStale(RuntimeError):
    pass


class HistoryImmutable(RuntimeError):
    pass


@dataclass(frozen=True)
class SnowSegment:
    """A taxiway segment to clear; lower priority values clear first."""
    segment_id: str
    priority: int
    clear_min: int
    snow_m3: float
    open_by: datetime | None = None

    def __post_init__(self):
        if not self.segment_id or self.clear_min <= 0 or self.snow_m3 <= 0:
            raise ValueError("invalid snow segment")
        if self.open_by is not None:
            object.__setattr__(self, "open_by", utc(self.open_by))


@dataclass(frozen=True)
class SweeperUnit:
    unit_id: str
    shift: Window
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.unit_id:
            raise ValueError("invalid sweeper unit")


@dataclass(frozen=True)
class DumpSite:
    site_id: str
    capacity_m3: float

    def __post_init__(self):
        if not self.site_id or self.capacity_m3 <= 0:
            raise ValueError("invalid dump site")


@dataclass(frozen=True)
class MovementBlock:
    """A committed aircraft movement that keeps a segment unavailable."""
    movement_id: str
    segment_id: str
    window: Window


@dataclass(frozen=True)
class SweepTask:
    task_id: str
    segment_id: str
    emergency: bool = False
    required: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.task_id or not self.segment_id:
            raise ValueError("invalid sweep task")


@dataclass(frozen=True)
class SweepAssignment:
    task_id: str
    segment_id: str
    unit_id: str
    dump_site_id: str
    window: Window
    snow_m3: float
    state: SweepState = SweepState.PLANNED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    version: int = 0

    def __post_init__(self):
        if self.started_at is not None:
            object.__setattr__(self, "started_at", utc(self.started_at))
        if self.completed_at is not None:
            object.__setattr__(self, "completed_at", utc(self.completed_at))


@dataclass(frozen=True)
class SnowPlan:
    generation: int
    considered: frozenset[str]
    proposed: tuple[SweepAssignment, ...]
    unscheduled: Mapping[str, str]
    openings: Mapping[str, datetime]


def blocks_from_routes(routes: Iterable[object]) -> tuple[MovementBlock, ...]:
    """Adapt committed surface routes (movement_id plus traversals) into movement blocks."""
    out = []
    for route in routes:
        for traversal in route.traversals:
            out.append(MovementBlock(route.movement_id, traversal.edge_id, traversal.window))
    return tuple(out)


class SnowOpsController:
    def __init__(
        self,
        segments: Iterable[SnowSegment],
        units: Iterable[SweeperUnit],
        dumps: Iterable[DumpSite],
        movements: Iterable[MovementBlock] = (),
    ):
        self._segments: dict[str, SnowSegment] = {}
        for segment in segments:
            if segment.segment_id in self._segments:
                raise ValueError(f"duplicate segment {segment.segment_id}")
            self._segments[segment.segment_id] = segment
        self._units: dict[str, SweeperUnit] = {}
        for unit in units:
            if unit.unit_id in self._units:
                raise ValueError(f"duplicate unit {unit.unit_id}")
            self._units[unit.unit_id] = unit
        self._dumps: dict[str, DumpSite] = {}
        for dump in dumps:
            if dump.site_id in self._dumps:
                raise ValueError(f"duplicate dump site {dump.site_id}")
            self._dumps[dump.site_id] = dump
        if not self._segments or not self._units or not self._dumps:
            raise ValueError("segments, sweeper units and dump sites are required")
        self._blocks: dict[str, list[Window]] = {sid: [] for sid in self._segments}
        self._tasks: dict[str, SweepTask] = {}
        self._assignments: dict[str, SweepAssignment] = {}
        self._cancelled: dict[str, SweepAssignment] = {}
        self._generation = 0
        self.add_movements(tuple(movements))

    @property
    def generation(self) -> int:
        return self._generation

    def request(self, task: SweepTask) -> None:
        if task.segment_id not in self._segments:
            raise ValueError(f"unknown segment {task.segment_id}")
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task {task.task_id}")
        self._tasks[task.task_id] = task

    def update_segment(self, segment: SnowSegment) -> None:
        if segment.segment_id not in self._segments:
            raise ValueError(f"unknown segment {segment.segment_id}")
        self._segments[segment.segment_id] = segment
        self._generation += 1

    def add_movements(self, blocks: Iterable[MovementBlock]) -> None:
        blocks = tuple(blocks)
        for block in blocks:
            if block.segment_id not in self._segments:
                raise ValueError(f"unknown segment {block.segment_id}")
        for block in blocks:
            self._blocks[block.segment_id].append(block.window)
        if blocks:
            self._generation += 1

    def assignment(self, task_id: str) -> SweepAssignment:
        if task_id in self._assignments:
            return self._assignments[task_id]
        if task_id in self._cancelled:
            return self._cancelled[task_id]
        raise KeyError(task_id)

    def assignments(self) -> tuple[SweepAssignment, ...]:
        return tuple(self._assignments[tid] for tid in sorted(self._assignments))

    def pending_tasks(self) -> tuple[SweepTask, ...]:
        return tuple(
            self._tasks[tid]
            for tid in sorted(self._tasks)
            if tid not in self._assignments and tid not in self._cancelled
        )

    def _replannable(self) -> tuple[str, ...]:
        return tuple(
            tid
            for tid in sorted(self._tasks)
            if tid not in self._cancelled
            and (tid not in self._assignments or self._assignments[tid].state is SweepState.PLANNED)
        )

    @staticmethod
    def _occupancy(assignment: SweepAssignment) -> Window:
        if (
            assignment.state is SweepState.COMPLETE
            and assignment.started_at is not None
            and assignment.completed_at is not None
        ):
            return Window(assignment.started_at, assignment.completed_at)
        if assignment.state is SweepState.EXECUTING and assignment.started_at is not None:
            end = max(assignment.window.end, assignment.started_at + timedelta(minutes=1))
            return Window(assignment.started_at, end)
        return assignment.window

    def _locked_view(self, replanned: frozenset[str]):
        kept = [
            a
            for a in self._assignments.values()
            if not (a.state is SweepState.PLANNED and a.task_id in replanned)
        ]
        unit_busy: dict[str, list[Window]] = {u: [] for u in self._units}
        seg_busy: dict[str, list[Window]] = {sid: list(ws) for sid, ws in self._blocks.items()}
        dump_used = {d: 0.0 for d in self._dumps}
        for a in kept:
            occupied = self._occupancy(a)
            unit_busy[a.unit_id].append(occupied)
            seg_busy[a.segment_id].append(occupied)
            dump_used[a.dump_site_id] += a.snow_m3
        return kept, unit_busy, seg_busy, dump_used

    @staticmethod
    def _earliest_window(
        busy: list[Window], duration_min: int, not_before: datetime, limit: datetime
    ) -> Window | None:
        start = not_before
        delta = timedelta(minutes=duration_min)
        for _ in range(1000):
            end = start + delta
            if end > limit:
                return None
            window = Window(start, end)
            hits = [b for b in busy if b.overlaps(window)]
            if not hits:
                return window
            start = max(b.end for b in hits)
        return None

    def _pick_dump(self, dump_used: Mapping[str, float], volume: float) -> str | None:
        fits = [
            (self._dumps[d].capacity_m3 - dump_used[d], d)
            for d in self._dumps
            if self._dumps[d].capacity_m3 - dump_used[d] >= volume - 1e-9
        ]
        if not fits:
            return None
        return min(fits)[1]

    @staticmethod
    def _project_openings(assignments: Iterable[SweepAssignment]) -> dict[str, datetime]:
        out: dict[str, datetime] = {}
        for a in assignments:
            opened = a.completed_at or a.window.end
            current = out.get(a.segment_id)
            if current is None or opened > current:
                out[a.segment_id] = opened
        return out

    def plan(self, at: datetime, task_ids: Iterable[str] | None = None) -> SnowPlan:
        at = utc(at)
        considered = set(self._replannable()) if task_ids is None else set(task_ids)
        for tid in considered:
            if tid not in self._tasks:
                raise ValueError(f"unknown task {tid}")
            if tid in self._cancelled:
                raise ValueError(f"task {tid} is cancelled")
            current = self._assignments.get(tid)
            if current is not None and current.state in LOCKED_STATES:
                raise ValueError(f"task {tid} already committed")
        replanned = frozenset(considered)
        kept, unit_busy, seg_busy, dump_used = self._locked_view(replanned)
        far = datetime.max.replace(tzinfo=UTC)
        order = sorted(
            considered,
            key=lambda tid: (
                not self._tasks[tid].emergency,
                self._segments[self._tasks[tid].segment_id].priority,
                self._segments[self._tasks[tid].segment_id].open_by or far,
                tid,
            ),
        )
        proposed = []
        unscheduled: dict[str, str] = {}
        for tid in order:
            task = self._tasks[tid]
            segment = self._segments[task.segment_id]
            best = None
            capable = False
            for unit in self._units.values():
                if not task.required.issubset(unit.capabilities):
                    continue
                capable = True
                start = max(at, unit.shift.start)
                busy = unit_busy[unit.unit_id] + seg_busy[segment.segment_id]
                window = self._earliest_window(busy, segment.clear_min, start, unit.shift.end)
                if window is None:
                    continue
                key = (window.end, window.start, unit.unit_id)
                if best is None or key < best[0]:
                    best = (key, unit, window)
            if best is None:
                unscheduled[tid] = "no-window" if capable else "no-capable-unit"
                continue
            site = self._pick_dump(dump_used, segment.snow_m3)
            if site is None:
                unscheduled[tid] = "dump-capacity"
                continue
            _, unit, window = best
            proposed.append(
                SweepAssignment(tid, segment.segment_id, unit.unit_id, site, window, segment.snow_m3)
            )
            unit_busy[unit.unit_id].append(window)
            seg_busy[segment.segment_id].append(window)
            dump_used[site] += segment.snow_m3
        openings = self._project_openings([*kept, *proposed])
        return SnowPlan(self._generation, replanned, tuple(proposed), unscheduled, openings)

    def commit(self, plan: SnowPlan) -> Decision:
        if plan.generation != self._generation:
            return Decision(
                False,
                "plan-stale",
                "snow plan computed against an older generation",
                {"plan_generation": plan.generation, "generation": self._generation},
            )
        _, unit_busy, seg_busy, dump_used = self._locked_view(plan.considered)
        seen = set()
        for proposal in plan.proposed:
            task = self._tasks.get(proposal.task_id)
            if task is None or task.segment_id != proposal.segment_id or proposal.task_id in self._cancelled:
                return Decision(False, "unknown-task", "proposal references an unknown task", {"task": proposal.task_id})
            if proposal.unit_id not in self._units or proposal.dump_site_id not in self._dumps:
                return Decision(False, "unknown-resource", "proposal references an unknown unit or dump site", {"task": proposal.task_id})
            if proposal.state is not SweepState.PLANNED:
                return Decision(False, "bad-state", "proposals must be unconfirmed", {"task": proposal.task_id})
            if proposal.task_id in seen:
                return Decision(False, "duplicate-proposal", "task proposed twice", {"task": proposal.task_id})
            seen.add(proposal.task_id)
            current = self._assignments.get(proposal.task_id)
            if current is not None and current.state in LOCKED_STATES:
                return Decision(False, "task-locked", "task already confirmed or executed", {"task": proposal.task_id})
            if any(w.overlaps(proposal.window) for w in unit_busy[proposal.unit_id]):
                return Decision(False, "unit-conflict", "sweeper already occupied", {"task": proposal.task_id, "unit": proposal.unit_id})
            if any(w.overlaps(proposal.window) for w in seg_busy[proposal.segment_id]):
                return Decision(False, "segment-conflict", "segment blocked by a committed movement or another sweep", {"task": proposal.task_id})
            if dump_used[proposal.dump_site_id] + proposal.snow_m3 > self._dumps[proposal.dump_site_id].capacity_m3 + 1e-9:
                return Decision(False, "dump-capacity", "dump site capacity exceeded", {"task": proposal.task_id})
            unit_busy[proposal.unit_id].append(proposal.window)
            seg_busy[proposal.segment_id].append(proposal.window)
            dump_used[proposal.dump_site_id] += proposal.snow_m3
        dropped = 0
        for tid in list(self._assignments):
            existing = self._assignments[tid]
            if existing.state is SweepState.PLANNED and tid in plan.considered:
                del self._assignments[tid]
                dropped += 1
        for proposal in plan.proposed:
            self._assignments[proposal.task_id] = proposal
        self._generation += 1
        return Decision(True, "ok", "snow plan committed", {"planned": len(plan.proposed), "dropped": dropped})

    def replan(self, at: datetime, task_ids: Iterable[str] | None = None) -> SnowPlan:
        plan = self.plan(at, task_ids)
        decision = self.commit(plan)
        if not decision.accepted:
            raise SnowPlanStale(f"replan rejected: {decision.code}")
        return plan

    def _require_active(self, task_id: str) -> SweepAssignment:
        assignment = self._assignments.get(task_id)
        if assignment is None:
            if task_id in self._cancelled:
                raise ValueError(f"task {task_id} is cancelled")
            raise KeyError(task_id)
        return assignment

    def confirm(self, task_id: str, at: datetime) -> SweepAssignment:
        assignment = self._require_active(task_id)
        if assignment.state is not SweepState.PLANNED:
            raise ValueError(f"cannot confirm from {assignment.state}")
        updated = replace(assignment, state=SweepState.CONFIRMED, version=assignment.version + 1)
        self._assignments[task_id] = updated
        self._generation += 1
        return updated

    def start(self, task_id: str, at: datetime) -> SweepAssignment:
        at = utc(at)
        assignment = self._require_active(task_id)
        if assignment.state is not SweepState.CONFIRMED:
            raise ValueError(f"cannot start from {assignment.state}")
        updated = replace(assignment, state=SweepState.EXECUTING, started_at=at, version=assignment.version + 1)
        self._assignments[task_id] = updated
        self._generation += 1
        return updated

    def complete(self, task_id: str, at: datetime) -> SweepAssignment:
        at = utc(at)
        assignment = self._require_active(task_id)
        if assignment.state is not SweepState.EXECUTING:
            raise ValueError(f"cannot complete from {assignment.state}")
        if assignment.started_at is not None and at <= assignment.started_at:
            raise ValueError("completion must follow start")
        updated = replace(assignment, state=SweepState.COMPLETE, completed_at=at, version=assignment.version + 1)
        self._assignments[task_id] = updated
        self._generation += 1
        return updated

    def cancel(self, task_id: str, at: datetime) -> SweepAssignment:
        assignment = self._require_active(task_id)
        if assignment.state in (SweepState.EXECUTING, SweepState.COMPLETE):
            raise HistoryImmutable(f"task {task_id} already executed; history cannot be rewritten")
        updated = replace(assignment, state=SweepState.CANCELLED, version=assignment.version + 1)
        del self._assignments[task_id]
        self._cancelled[task_id] = updated
        self._generation += 1
        return updated

    def openings(self) -> Mapping[str, datetime]:
        return self._project_openings(self._assignments.values())

    def dump_remaining(self) -> Mapping[str, float]:
        used = {d: 0.0 for d in self._dumps}
        for a in self._assignments.values():
            used[a.dump_site_id] += a.snow_m3
        return {d: self._dumps[d].capacity_m3 - used[d] for d in self._dumps}

    def conflicts(self) -> tuple[Mapping[str, str], ...]:
        out = []
        for a in sorted(self._assignments.values(), key=lambda x: x.task_id):
            if a.state not in LOCKED_STATES:
                continue
            if any(w.overlaps(a.window) for w in self._blocks.get(a.segment_id, ())):
                out.append({"task_id": a.task_id, "code": "movement-conflict", "segment_id": a.segment_id})
        return tuple(out)

    def history(self) -> tuple[SweepAssignment, ...]:
        rows = [
            a
            for a in self._assignments.values()
            if a.state in (SweepState.EXECUTING, SweepState.COMPLETE)
        ]
        return tuple(sorted(rows, key=lambda a: (a.started_at or a.window.start, a.task_id)))

    def history_digest(self) -> str:
        payload = [
            {
                "task_id": a.task_id,
                "segment_id": a.segment_id,
                "unit_id": a.unit_id,
                "dump_site_id": a.dump_site_id,
                "window": [a.window.start.isoformat(), a.window.end.isoformat()],
                "state": a.state.value,
                "started_at": a.started_at.isoformat() if a.started_at else None,
                "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                "version": a.version,
            }
            for a in self.history()
        ]
        raw = dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return sha256(raw).hexdigest()
