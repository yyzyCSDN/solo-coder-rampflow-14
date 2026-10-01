import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from rampflow.timeutil import Window
from rampflow.snow import (
    DumpSite,
    HistoryImmutable,
    MovementBlock,
    SnowOpsController,
    SnowSegment,
    SweepState,
    SweepTask,
    SweeperUnit,
    blocks_from_routes,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 2, 1, h, m, tzinfo=UTC)


class SnowOpsTests(unittest.TestCase):
    def test_priority_order_and_openings(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 30, 40), SnowSegment("B", 20, 30, 40), SnowSegment("C", 30, 30, 40)],
            [SweeperUnit("U1", Window(D(6), D(18))), SweeperUnit("U2", Window(D(6), D(18)))],
            [DumpSite("NORTH", 1000)],
        )
        for tid, seg in (("T-A", "A"), ("T-B", "B"), ("T-C", "C")):
            ctl.request(SweepTask(tid, seg))
        plan = ctl.replan(D(6))
        by_id = {a.task_id: a for a in plan.proposed}
        self.assertEqual(by_id["T-A"].window, Window(D(6), D(6, 30)))
        self.assertEqual(by_id["T-A"].unit_id, "U1")
        self.assertEqual(by_id["T-B"].window, Window(D(6), D(6, 30)))
        self.assertEqual(by_id["T-B"].unit_id, "U2")
        self.assertEqual(by_id["T-C"].window, Window(D(6, 30), D(7)))
        openings = ctl.openings()
        self.assertEqual(openings["A"], D(6, 30))
        self.assertEqual(openings["B"], D(6, 30))
        self.assertEqual(openings["C"], D(7))

    def test_committed_movements_are_avoided(self):
        movements = [MovementBlock("M1", "A", Window(D(6), D(6, 45)))]
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 30, 40)],
            [SweeperUnit("U1", Window(D(6), D(18)))],
            [DumpSite("NORTH", 1000)],
            movements,
        )
        ctl.request(SweepTask("T-A", "A"))
        plan = ctl.replan(D(6))
        assignment = plan.proposed[0]
        self.assertEqual(assignment.window, Window(D(6, 45), D(7, 15)))
        self.assertFalse(assignment.window.overlaps(Window(D(6), D(6, 45))))

    def test_dump_capacity_limits_schedule(self):
        segments = [SnowSegment(s, p, 30, 50) for s, p in (("A", 10), ("B", 20), ("C", 30), ("E", 40))]
        ctl = SnowOpsController(
            segments,
            [SweeperUnit("U1", Window(D(6), D(18))), SweeperUnit("U2", Window(D(6), D(18)))],
            [DumpSite("NORTH", 100), DumpSite("SOUTH", 50)],
        )
        for seg in ("A", "B", "C", "E"):
            ctl.request(SweepTask(f"T-{seg}", seg))
        plan = ctl.replan(D(6))
        by_id = {a.task_id: a for a in plan.proposed}
        self.assertEqual(by_id["T-A"].dump_site_id, "SOUTH")
        self.assertEqual(by_id["T-B"].dump_site_id, "NORTH")
        self.assertEqual(by_id["T-C"].dump_site_id, "NORTH")
        self.assertEqual(plan.unscheduled, {"T-E": "dump-capacity"})
        remaining = ctl.dump_remaining()
        self.assertEqual(remaining["NORTH"], 0)
        self.assertEqual(remaining["SOUTH"], 0)

    def test_emergency_preempts_unconfirmed(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 60, 10), SnowSegment("B", 20, 60, 10), SnowSegment("E", 5, 60, 10)],
            [SweeperUnit("U1", Window(D(6), D(8)))],
            [DumpSite("NORTH", 1000)],
        )
        ctl.request(SweepTask("T-A", "A"))
        ctl.request(SweepTask("T-B", "B"))
        ctl.replan(D(6))
        ctl.confirm("T-A", D(6))
        ctl.request(SweepTask("T-E", "E", emergency=True))
        plan = ctl.replan(D(6))
        by_id = {a.task_id: a for a in plan.proposed}
        self.assertEqual(by_id["T-E"].window, Window(D(7), D(8)))
        self.assertEqual(plan.unscheduled.get("T-B"), "no-window")
        confirmed = ctl.assignment("T-A")
        self.assertEqual(confirmed.window, Window(D(6), D(7)))
        self.assertEqual(confirmed.state, SweepState.CONFIRMED)
        self.assertNotIn("T-B", {a.task_id for a in ctl.assignments()})
        self.assertIn("T-B", {t.task_id for t in ctl.pending_tasks()})

    def test_emergency_cannot_move_confirmed(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 60, 10), SnowSegment("B", 20, 60, 10), SnowSegment("E", 5, 60, 10)],
            [SweeperUnit("U1", Window(D(6), D(8)))],
            [DumpSite("NORTH", 1000)],
        )
        ctl.request(SweepTask("T-A", "A"))
        ctl.request(SweepTask("T-B", "B"))
        ctl.replan(D(6))
        ctl.confirm("T-A", D(6))
        ctl.confirm("T-B", D(6))
        ctl.request(SweepTask("T-E", "E", emergency=True))
        plan = ctl.replan(D(6))
        self.assertEqual(plan.unscheduled.get("T-E"), "no-window")
        self.assertEqual(ctl.assignment("T-A").window, Window(D(6), D(7)))
        self.assertEqual(ctl.assignment("T-B").window, Window(D(7), D(8)))

    def test_executed_history_is_immutable(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 60, 10), SnowSegment("B", 20, 60, 10)],
            [SweeperUnit("U1", Window(D(6), D(18)))],
            [DumpSite("NORTH", 1000)],
        )
        ctl.request(SweepTask("T-A", "A"))
        ctl.replan(D(6))
        ctl.confirm("T-A", D(6))
        ctl.start("T-A", D(6, 5))
        with self.assertRaises(HistoryImmutable):
            ctl.cancel("T-A", D(6, 10))
        ctl.complete("T-A", D(7, 5))
        with self.assertRaises(ValueError):
            ctl.complete("T-A", D(8))
        history_before = ctl.history()
        digest_before = ctl.history_digest()
        ctl.request(SweepTask("T-B", "B"))
        plan = ctl.replan(D(7, 10))
        self.assertEqual(ctl.history(), history_before)
        self.assertEqual(ctl.history_digest(), digest_before)
        executed = ctl.history()[0]
        self.assertEqual(executed.window, Window(D(6), D(7)))
        self.assertEqual(executed.started_at, D(6, 5))
        self.assertEqual(executed.completed_at, D(7, 5))
        tb = {a.task_id: a for a in plan.proposed}["T-B"]
        self.assertEqual(tb.window, Window(D(7, 10), D(8, 10)))
        with self.assertRaises(ValueError):
            ctl.start("T-B", D(7, 30))

    def test_stale_plan_rejected(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 30, 10)],
            [SweeperUnit("U1", Window(D(6), D(18)))],
            [DumpSite("NORTH", 100)],
        )
        ctl.request(SweepTask("T-A", "A"))
        plan = ctl.plan(D(6))
        ctl.add_movements([MovementBlock("M1", "A", Window(D(6), D(7)))])
        decision = ctl.commit(plan)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "plan-stale")

    def test_conflicts_reports_movement_overlap(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 30, 10)],
            [SweeperUnit("U1", Window(D(6), D(18)))],
            [DumpSite("NORTH", 100)],
        )
        ctl.request(SweepTask("T-A", "A"))
        ctl.replan(D(6))
        ctl.confirm("T-A", D(6))
        self.assertEqual(ctl.conflicts(), ())
        ctl.add_movements([MovementBlock("M9", "A", Window(D(6, 15), D(6, 45)))])
        conflicts = ctl.conflicts()
        self.assertEqual([c["task_id"] for c in conflicts], ["T-A"])

    def test_cancel_releases_dump_reservation(self):
        ctl = SnowOpsController(
            [SnowSegment("A", 10, 30, 50)],
            [SweeperUnit("U1", Window(D(6), D(18)))],
            [DumpSite("NORTH", 60)],
        )
        ctl.request(SweepTask("T-A", "A"))
        ctl.replan(D(6))
        ctl.confirm("T-A", D(6))
        self.assertEqual(ctl.dump_remaining()["NORTH"], 10)
        ctl.cancel("T-A", D(6, 5))
        self.assertEqual(ctl.dump_remaining()["NORTH"], 60)

    def test_blocks_from_routes(self):
        route = SimpleNamespace(
            movement_id="M1",
            traversals=[
                SimpleNamespace(edge_id="A", window=Window(D(6), D(6, 5))),
                SimpleNamespace(edge_id="B", window=Window(D(6, 5), D(6, 10))),
            ],
        )
        blocks = blocks_from_routes([route])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].segment_id, "A")
        self.assertEqual(blocks[0].movement_id, "M1")
        self.assertEqual(blocks[1].window, Window(D(6, 5), D(6, 10)))


if __name__ == "__main__":
    unittest.main()
