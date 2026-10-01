import unittest
from datetime import datetime, timezone

from rampflow.timeutil import Window
from rampflow.surface_movement import SurfaceRouter, TaxiEdge
from rampflow.snow_removal import (
    ClearanceState,
    SnowConflict,
    SnowDump,
    SnowPlanner,
    SnowSegment,
    SweeperUnit,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def make_planner(moves=None, units=None, dumps=None):
    units = units or [SweeperUnit("U1", Window(D(6), D(18))), SweeperUnit("U2", Window(D(6), D(18)))]
    dumps = dumps or [SnowDump("NEAR", 500, 5), SnowDump("FAR", 2000, 25)]
    provider = (lambda sid: moves.get(sid, ())) if moves is not None else None
    return SnowPlanner(units, dumps, movement_windows=provider)


def by_segment(planner):
    return {a.segment_id: a for a in planner.assignments}


class SnowRemovalTests(unittest.TestCase):
    def test_priority_orders_fleet_and_reopening(self):
        p = make_planner()
        segs = [
            SnowSegment("TWY-B", 30, 100, 20),
            SnowSegment("RWY", 10, 300, 30),
            SnowSegment("TWY-A", 20, 200, 20),
        ]
        plan = p.plan(segs, D(7))
        self.assertTrue(p.commit(plan).accepted)
        amap = by_segment(p)
        self.assertEqual(amap["RWY"].window, Window(D(7), D(7, 35)))
        self.assertEqual(amap["RWY"].unit_id, "U1")
        self.assertEqual(amap["TWY-A"].window, Window(D(7), D(7, 25)))
        reopen = p.reopening()
        self.assertEqual(reopen, (("TWY-A", D(7, 25)), ("RWY", D(7, 35)), ("TWY-B", D(8, 10))))
        for unit in ("U1", "U2"):
            ws = sorted(a.window for a in p.assignments if a.unit_id == unit)
            for first, second in zip(ws, ws[1:]):
                self.assertLessEqual(first.end, second.start)

    def test_dump_capacity_routes_snow(self):
        p = make_planner()
        segs = [SnowSegment("RWY", 10, 300, 30), SnowSegment("TWY-A", 20, 200, 20), SnowSegment("TWY-B", 30, 100, 20)]
        p.commit(p.plan(segs, D(7)))
        amap = by_segment(p)
        self.assertEqual(amap["RWY"].dump_id, "NEAR")
        self.assertEqual(amap["TWY-B"].dump_id, "FAR")  # NEAR full, overflow hauls farther
        loads = p.dump_loads()
        self.assertEqual((loads["NEAR"], loads["FAR"]), (500.0, 100.0))
        with self.assertRaises(SnowConflict):
            p.plan([SnowSegment("HUGE", 40, 5000, 30)], D(7))

    def test_committed_aircraft_movements_are_avoided(self):
        router = SurfaceRouter([TaxiEdge("TWY-A", "S1", "S2", 600)], headway_seconds=0, crossing_headway_seconds=0)
        route = router.plan("MV1", "S1", "S2", D(8), 30.0)
        router.commit(route)
        moves = {}
        for tr in route.traversals:
            moves.setdefault(tr.edge_id, []).append(tr.window)
        p = make_planner(moves=moves, units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=[SnowDump("NEAR", 1000, 5)])
        plan = p.plan([SnowSegment("TWY-A", 10, 100, 20)], D(7, 50))
        self.assertTrue(p.commit(plan).accepted)
        self.assertEqual(p.assignments[0].window, Window(D(8, 10), D(8, 35)))

    def test_new_movement_between_plan_and_commit_rejects(self):
        moves = {}
        p = make_planner(moves=moves, units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=[SnowDump("NEAR", 1000, 5)])
        plan = p.plan([SnowSegment("TWY-A", 10, 100, 20)], D(7))
        moves["TWY-A"] = [Window(D(7, 10), D(7, 20))]
        dec = p.commit(plan)
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.code, "movement-conflict")
        self.assertEqual(p.assignments, ())

    def test_stale_plan_rejected(self):
        p = make_planner()
        plan1 = p.plan([SnowSegment("A", 10, 100, 20)], D(7))
        plan2 = p.plan([SnowSegment("B", 20, 100, 20)], D(7))
        self.assertTrue(p.commit(plan1).accepted)
        dec = p.commit(plan2)
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.code, "stale-plan")

    def test_emergency_preempts_only_tentative(self):
        p = make_planner(units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=[SnowDump("NEAR", 1000, 5)])
        p.commit(p.plan([SnowSegment("A", 10, 100, 60), SnowSegment("B", 20, 100, 60)], D(7)))
        p.confirm(["A"])
        dec = p.preempt(SnowSegment("E", 0, 50, 30), D(7, 30))
        self.assertTrue(dec.accepted)
        self.assertEqual(dec.details["displaced"], ("B",))
        amap = by_segment(p)
        self.assertEqual(amap["A"].window, Window(D(7), D(8, 5)))  # committed work unmoved
        self.assertEqual(amap["A"].state, ClearanceState.COMMITTED)
        self.assertEqual(amap["E"].window, Window(D(8, 5), D(8, 40)))
        self.assertEqual(amap["E"].state, ClearanceState.COMMITTED)
        self.assertEqual(amap["B"].window, Window(D(8, 40), D(9, 45)))  # tentative work bumped
        self.assertEqual(amap["B"].state, ClearanceState.PLANNED)
        self.assertEqual(p.history, ())

    def test_emergency_without_slot_leaves_state_untouched(self):
        p = make_planner(units=[SweeperUnit("U1", Window(D(7), D(9)))], dumps=[SnowDump("NEAR", 1000, 5)])
        p.commit(p.plan([SnowSegment("A", 10, 100, 60)], D(7)))
        p.confirm(["A"])
        before = (p.assignments, p.generation)
        dec = p.preempt(SnowSegment("E", 0, 50, 60), D(8))
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.code, "no-emergency-slot")
        self.assertEqual((p.assignments, p.generation), before)

    def test_preempt_rolls_back_when_displaced_work_cannot_fit(self):
        dumps = [SnowDump("NEAR", 150, 5), SnowDump("FAR", 60, 25)]
        p = make_planner(units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=dumps)
        p.commit(p.plan([SnowSegment("A", 10, 50, 30), SnowSegment("B", 20, 100, 30)], D(7)))
        before = (p.assignments, p.generation)
        dec = p.preempt(SnowSegment("E", 0, 100, 30), D(7))
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.code, "preempt-infeasible")
        self.assertEqual((p.assignments, p.generation), before)

    def test_executed_history_is_immutable(self):
        p = make_planner(units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=[SnowDump("NEAR", 1000, 5)])
        p.commit(p.plan([SnowSegment("A", 10, 100, 60), SnowSegment("B", 20, 100, 60)], D(7)))
        p.confirm()
        rec = p.record_execution("A", Window(D(7), D(7, 40)))
        self.assertEqual(rec.state, ClearanceState.EXECUTED)
        self.assertNotIn("A", by_segment(p))
        with self.assertRaises(SnowConflict):  # unit cannot have worked two places at once
            p.record_execution("B", Window(D(7, 30), D(8, 30)))
        p.record_execution("B", Window(D(7, 40), D(8, 50)))
        hist = p.history
        p.replan(D(9))
        self.assertEqual(p.history, hist)  # replanning never rewrites the past
        with self.assertRaises(SnowConflict):
            p.record_execution("A", Window(D(10), D(11)))
        self.assertEqual(p.reopening(), (("A", D(7, 40)), ("B", D(8, 50))))

    def test_executed_snow_counts_against_dump_capacity(self):
        dumps = [SnowDump("NEAR", 250, 5), SnowDump("FAR", 250, 25)]
        p = make_planner(units=[SweeperUnit("U1", Window(D(6), D(18)))], dumps=dumps)
        p.commit(p.plan([SnowSegment("A", 10, 200, 30), SnowSegment("B", 20, 100, 30)], D(7)))
        p.confirm()
        p.record_execution("A", Window(D(7), D(7, 40)))
        with self.assertRaises(SnowConflict):
            p.record_execution("B", Window(D(8), D(8, 40)), dump_id="NEAR")
        p.record_execution("B", Window(D(8), D(8, 40)))
        self.assertEqual(p.dump_loads()["FAR"], 100.0)

    def test_replan_keeps_committed_fixed(self):
        p = make_planner(dumps=[SnowDump("NEAR", 1000, 5)])
        segs = [SnowSegment("A", 10, 100, 60), SnowSegment("B", 20, 100, 60), SnowSegment("C", 30, 100, 60)]
        p.commit(p.plan(segs, D(7)))
        p.confirm(["A"])
        before = by_segment(p)
        p.replan(D(8, 30))
        after = by_segment(p)
        self.assertEqual(after["A"], before["A"])
        self.assertGreaterEqual(after["B"].window.start, D(8, 30))
        self.assertGreaterEqual(after["C"].window.start, D(8, 30))
        self.assertEqual(p.history, ())

    def test_dependencies_respected_despite_priority(self):
        p = make_planner()
        segs = [SnowSegment("B", 5, 100, 30, frozenset({"A"})), SnowSegment("A", 10, 100, 30)]
        p.commit(p.plan(segs, D(7)))
        amap = by_segment(p)
        self.assertGreaterEqual(amap["B"].window.start, amap["A"].window.end)
        with self.assertRaises(ValueError):
            p.plan([SnowSegment("X", 1, 100, 30, frozenset({"Y"})), SnowSegment("Y", 1, 100, 30, frozenset({"X"}))], D(9))
        with self.assertRaises(ValueError):
            p.plan([SnowSegment("Z", 1, 100, 30, frozenset({"NOPE"}))], D(9))

    def test_release_only_tentative_and_without_dependents(self):
        p = make_planner()
        segs = [SnowSegment("A", 10, 100, 30), SnowSegment("B", 20, 100, 30, frozenset({"A"}))]
        p.commit(p.plan(segs, D(7)))
        with self.assertRaises(SnowConflict):
            p.release("A")
        p.release("B")
        self.assertNotIn("B", by_segment(p))
        p.confirm(["A"])
        with self.assertRaises(SnowConflict):
            p.release("A")
        with self.assertRaises(KeyError):
            p.confirm(["NOPE"])


if __name__ == "__main__":
    unittest.main()
