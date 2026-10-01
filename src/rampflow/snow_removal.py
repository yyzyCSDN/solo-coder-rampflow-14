from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from enum import Enum
from typing import Callable,Iterable
from .timeutil import Window,utc
from .models import Decision

class ClearanceState(str,Enum):
    PLANNED="planned"
    COMMITTED="committed"
    EXECUTED="executed"

@dataclass(frozen=True)
class SnowSegment:
    segment_id:str
    priority:int
    snow_m3:float
    clear_min:int
    dependencies:frozenset[str]=frozenset()
    def __post_init__(self):
        if not self.segment_id:raise ValueError("segment id required")
        if self.snow_m3<=0:raise ValueError("positive snow volume required")
        if self.clear_min<=0:raise ValueError("positive clear duration required")
        if self.segment_id in self.dependencies:raise ValueError("self dependency")

@dataclass(frozen=True)
class SnowDump:
    dump_id:str
    capacity_m3:float
    haul_min:int=8
    def __post_init__(self):
        if not self.dump_id:raise ValueError("dump id required")
        if self.capacity_m3<=0:raise ValueError("positive dump capacity required")
        if self.haul_min<0:raise ValueError("negative haul time")

@dataclass(frozen=True)
class SweeperUnit:
    unit_id:str
    shift:Window
    def __post_init__(self):
        if not self.unit_id:raise ValueError("unit id required")

@dataclass(frozen=True)
class ClearanceAssignment:
    segment_id:str
    unit_id:str
    dump_id:str
    window:Window
    snow_m3:float
    state:ClearanceState
    version:int=0

@dataclass(frozen=True)
class SnowPlan:
    generation:int
    segments:tuple[SnowSegment,...]
    assignments:tuple[ClearanceAssignment,...]
    reopening:tuple[tuple[str,datetime],...]

class SnowConflict(RuntimeError):pass

def _reopening_of(assignments:Iterable[ClearanceAssignment])->tuple[tuple[str,datetime],...]:
    return tuple(sorted(((a.segment_id,a.window.end) for a in assignments),key=lambda x:(x[1],x[0])))

class SnowPlanner:
    def __init__(self,units:Iterable[SweeperUnit],dumps:Iterable[SnowDump],movement_windows:Callable[[str],Iterable[Window]]|None=None):
        self.units={}
        for u in units:
            if u.unit_id in self.units:raise ValueError(f"duplicate unit {u.unit_id}")
            self.units[u.unit_id]=u
        self.dumps={}
        for d in dumps:
            if d.dump_id in self.dumps:raise ValueError(f"duplicate dump {d.dump_id}")
            self.dumps[d.dump_id]=d
        if not self.units:raise ValueError("at least one sweeper unit required")
        if not self.dumps:raise ValueError("at least one snow dump required")
        self._movements=movement_windows or (lambda segment_id:())
        self._segments:dict[str,SnowSegment]={}
        self._assignments:dict[str,ClearanceAssignment]={}
        self._history:list[ClearanceAssignment]=[]
        self._generation=0

    @property
    def generation(self):return self._generation

    @property
    def history(self):return tuple(self._history)

    @property
    def assignments(self):
        return tuple(sorted(self._assignments.values(),key=lambda a:(a.window.start,a.segment_id)))

    def dump_loads(self):return self._dump_loads()

    def reopening(self):
        items=[(a.segment_id,a.window.end) for a in self._assignments.values()]
        items.extend((h.segment_id,h.window.end) for h in self._history)
        return tuple(sorted(items,key=lambda x:(x[1],x[0])))

    def _dump_loads(self,extra:Iterable[ClearanceAssignment]=()):
        loads={d:0.0 for d in self.dumps}
        for a in tuple(self._assignments.values())+tuple(self._history)+tuple(extra):
            loads[a.dump_id]+=a.snow_m3
        return loads

    def _fixed(self):
        busy={u:[] for u in self.units}
        for a in self._history:busy[a.unit_id].append(a.window)
        for a in self._assignments.values():busy[a.unit_id].append(a.window)
        return busy

    def _completion(self):
        done={a.segment_id:a.window.end for a in self._history}
        done.update({a.segment_id:a.window.end for a in self._assignments.values()})
        return done

    def _successor_deadlines(self,moving:set[str]):
        # fixed work depending on a still-movable segment pins its latest finish
        deadlines={}
        fixed={a.segment_id:a for a in self._history}
        fixed.update(self._assignments)
        for sid,a in fixed.items():
            seg=self._segments.get(sid)
            if seg is None:continue
            for dep in seg.dependencies:
                if dep in moving:deadlines[dep]=min(deadlines.get(dep,a.window.start),a.window.start)
        return deadlines

    def _fit(self,segment:SnowSegment,unit:SweeperUnit,dump:SnowDump,earliest:datetime,latest:datetime|None,busy:list[Window]):
        duration=timedelta(minutes=segment.clear_min+dump.haul_min)
        blocked=[w for w in self._movements(segment.segment_id)]
        t=max(utc(earliest),unit.shift.start)
        for _ in range(500):
            w=Window(t,t+duration)
            if w.end>unit.shift.end:return None
            if latest is not None and w.end>latest:return None
            clashes=[b for b in busy if b.overlaps(w)]
            clashes.extend(b for b in blocked if b.overlaps(w))
            if not clashes:return w
            t=max(b.end for b in clashes)
        return None

    def _place(self,segment:SnowSegment,earliest:datetime,latest:datetime|None,busy:dict[str,list[Window]],loads:dict[str,float],state:ClearanceState=ClearanceState.PLANNED):
        best=None
        for uid in sorted(self.units):
            for did in sorted(self.dumps):
                dump=self.dumps[did]
                if loads[did]+segment.snow_m3>dump.capacity_m3:continue
                w=self._fit(segment,self.units[uid],dump,earliest,latest,busy[uid])
                if w is None:continue
                key=(w.end,w.start,uid,did)
                if best is None or key<best[0]:best=(key,uid,did,w)
        if best is None:return None
        _,uid,did,w=best
        return ClearanceAssignment(segment.segment_id,uid,did,w,segment.snow_m3,state)

    def _schedule(self,segments:tuple[SnowSegment,...],start:datetime):
        segs={s.segment_id:s for s in segments}
        completion=self._completion()
        for s in segs.values():
            missing=set(s.dependencies)-set(segs)-set(completion)
            if missing:raise ValueError(f"missing dependency for {s.segment_id}: {sorted(missing)}")
        remaining={k:set(v.dependencies)&set(segs) for k,v in segs.items()}
        busy=self._fixed();loads=self._dump_loads();deadlines=self._successor_deadlines(set(segs))
        placed=[];ready=[sid for sid,d in remaining.items() if not d]
        while ready:
            ready.sort(key=lambda x:(segs[x].priority,x))
            sid=ready.pop(0);s=segs[sid]
            earliest=max([utc(start)]+[completion[d] for d in s.dependencies])
            a=self._place(s,earliest,deadlines.get(sid),busy,loads)
            if a is None:raise SnowConflict(f"no clearing capacity for {sid}")
            busy[a.unit_id].append(a.window);loads[a.dump_id]+=s.snow_m3;completion[sid]=a.window.end
            placed.append(a)
            for other,deps in remaining.items():
                if sid in deps:
                    deps.remove(sid)
                    if not deps:ready.append(other)
        if len(placed)!=len(segs):raise ValueError("clearing dependency cycle")
        return tuple(placed)

    def plan(self,segments:Iterable[SnowSegment],start:datetime)->SnowPlan:
        segments=tuple(segments)
        ids=[s.segment_id for s in segments]
        if len(set(ids))!=len(ids):raise ValueError("duplicate segment in batch")
        for s in segments:
            if s.segment_id in self._segments:raise ValueError(f"segment already known: {s.segment_id}")
        placed=self._schedule(segments,start)
        return SnowPlan(self._generation,segments,placed,_reopening_of(placed))

    def commit(self,plan:SnowPlan)->Decision:
        if plan.generation!=self._generation:
            return Decision(False,"stale-plan","planner state changed since plan was computed",{"generation":self._generation})
        busy=self._fixed();loads=self._dump_loads();staged=[]
        for a in plan.assignments:
            if a.segment_id in self._segments:
                return Decision(False,"segment-known",f"segment already arranged: {a.segment_id}",{"segment":a.segment_id})
            unit=self.units.get(a.unit_id);dump=self.dumps.get(a.dump_id)
            if unit is None or dump is None:
                return Decision(False,"unknown-resource","plan references unknown unit or dump",{"segment":a.segment_id})
            if a.window.start<unit.shift.start or a.window.end>unit.shift.end:
                return Decision(False,"shift-violation",f"window outside shift of {a.unit_id}",{"segment":a.segment_id})
            if any(w.overlaps(a.window) for w in busy[a.unit_id]):
                return Decision(False,"unit-conflict",f"unit {a.unit_id} no longer free",{"segment":a.segment_id})
            if any(w.overlaps(a.window) for w in self._movements(a.segment_id)):
                return Decision(False,"movement-conflict","committed aircraft movement blocks the window",{"segment":a.segment_id})
            if loads[a.dump_id]+a.snow_m3>dump.capacity_m3:
                return Decision(False,"dump-overflow",f"dump {a.dump_id} capacity exceeded",{"segment":a.segment_id})
            busy[a.unit_id].append(a.window);loads[a.dump_id]+=a.snow_m3;staged.append(a)
        for s in plan.segments:self._segments[s.segment_id]=s
        for a in staged:self._assignments[a.segment_id]=a
        self._generation+=1
        return Decision(True,"ok","snow clearance plan committed",{"count":len(staged)})

    def confirm(self,segment_ids:Iterable[str]|None=None)->tuple[ClearanceAssignment,...]:
        ids=sorted(set(segment_ids)) if segment_ids is not None else sorted(self._assignments)
        missing=[s for s in ids if s not in self._assignments]
        if missing:raise KeyError(f"unknown segments: {missing}")
        out=[];changed=False
        for sid in ids:
            a=self._assignments[sid]
            if a.state==ClearanceState.PLANNED:
                a=replace(a,state=ClearanceState.COMMITTED,version=a.version+1)
                self._assignments[sid]=a;changed=True
            out.append(a)
        if changed:self._generation+=1
        return tuple(out)

    def release(self,segment_id:str)->ClearanceAssignment:
        a=self._assignments.get(segment_id)
        if a is None:raise KeyError(f"unknown segment: {segment_id}")
        if a.state!=ClearanceState.PLANNED:raise SnowConflict("only tentative assignments can be released")
        dependents=sorted(s.segment_id for s in self._segments.values() if segment_id in s.dependencies)
        if dependents:raise SnowConflict(f"segment {segment_id} still required by {dependents}")
        del self._assignments[segment_id];del self._segments[segment_id]
        self._generation+=1
        return a

    def replan(self,start:datetime)->SnowPlan:
        start=utc(start)
        moving={sid:a for sid,a in self._assignments.items() if a.state==ClearanceState.PLANNED}
        saved=self._assignments
        self._assignments={sid:a for sid,a in self._assignments.items() if a.state==ClearanceState.COMMITTED}
        try:
            placed=self._schedule(tuple(self._segments[sid] for sid in moving),start)
        except (SnowConflict,ValueError):
            self._assignments=saved
            raise
        for a in placed:self._assignments[a.segment_id]=a
        self._generation+=1
        segments=tuple(self._segments[sid] for sid in moving)
        return SnowPlan(self._generation,segments,placed,_reopening_of(placed))

    def preempt(self,segment:SnowSegment,start:datetime)->Decision:
        start=utc(start)
        if segment.segment_id in self._segments:raise ValueError(f"segment already known: {segment.segment_id}")
        firm={a.segment_id:a.window.end for a in self._history}
        firm.update({a.segment_id:a.window.end for a in self._assignments.values() if a.state==ClearanceState.COMMITTED})
        missing=set(segment.dependencies)-set(firm)
        if missing:raise ValueError(f"missing dependency for {segment.segment_id}: {sorted(missing)}")
        earliest=max([start]+[firm[d] for d in segment.dependencies])
        saved_assignments=self._assignments;saved_segments=self._segments
        bumped={sid:a for sid,a in self._assignments.items() if a.state==ClearanceState.PLANNED}
        # tentative work yields: the emergency is placed against committed and executed work only
        self._assignments={sid:a for sid,a in self._assignments.items() if a.state==ClearanceState.COMMITTED}
        placed=self._place(segment,earliest,None,self._fixed(),self._dump_loads(),state=ClearanceState.COMMITTED)
        if placed is None:
            self._assignments=saved_assignments
            return Decision(False,"no-emergency-slot","emergency clearing infeasible without moving committed or executed work",{"segment":segment.segment_id})
        self._assignments[segment.segment_id]=placed
        self._segments=dict(self._segments);self._segments[segment.segment_id]=segment
        try:
            replanned=self._schedule(tuple(saved_segments[sid] for sid in bumped),start)
        except (SnowConflict,ValueError) as error:
            self._assignments=saved_assignments;self._segments=saved_segments
            return Decision(False,"preempt-infeasible",f"emergency fits but displaced work cannot be re-placed: {error}",{"segment":segment.segment_id})
        for a in replanned:self._assignments[a.segment_id]=a
        self._generation+=1
        return Decision(True,"ok","emergency clearing committed; tentative work re-placed",{"segment":segment.segment_id,"unit":placed.unit_id,"dump":placed.dump_id,"window":placed.window,"displaced":tuple(sorted(bumped))})

    def record_execution(self,segment_id:str,window:Window,unit_id:str|None=None,dump_id:str|None=None)->ClearanceAssignment:
        a=self._assignments.get(segment_id)
        if a is None:
            if any(h.segment_id==segment_id for h in self._history):raise SnowConflict(f"segment already executed: {segment_id}")
            raise KeyError(f"unknown segment: {segment_id}")
        unit_id=unit_id or a.unit_id;dump_id=dump_id or a.dump_id
        if unit_id not in self.units:raise ValueError(f"unknown unit {unit_id}")
        if dump_id not in self.dumps:raise ValueError(f"unknown dump {dump_id}")
        # executed routes are append-only history and must stay mutually consistent
        if any(h.unit_id==unit_id and h.window.overlaps(window) for h in self._history):
            raise SnowConflict("executed history conflict: unit already worked that window")
        snow=self._segments[segment_id].snow_m3
        loads=self._dump_loads()
        effective=loads[dump_id]-(a.snow_m3 if a.dump_id==dump_id else 0.0)+snow
        if effective>self.dumps[dump_id].capacity_m3:raise SnowConflict(f"dump {dump_id} capacity exceeded")
        rec=ClearanceAssignment(segment_id,unit_id,dump_id,window,snow,ClearanceState.EXECUTED,a.version+1)
        del self._assignments[segment_id]
        self._history.append(rec)
        self._generation+=1
        return rec
