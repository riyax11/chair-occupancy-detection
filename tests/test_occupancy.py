import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from occupancy import (  # noqa: E402
    OccupancyTracker,
    assign_people_to_chairs,
    match_people_to_chairs,
    overlap_ratio,
    posture,
)

CHAIRS = {1: (100, 100, 160, 160), 2: (200, 100, 260, 160)}
SEATED = (90, 60, 170, 160)      # 80 wide x 100 tall: shorter, wide shape
STANDING = (110, 0, 150, 160)    # 40 wide x 160 tall: tall and narrow


class PostureTests(unittest.TestCase):
    def test_seated_and_standing_shapes(self):
        self.assertEqual(posture(SEATED), "sitting")
        self.assertEqual(posture(STANDING), "standing")

    def test_threshold_is_tunable(self):
        person = (0, 0, 50, 100)  # height / width = 2.0
        self.assertEqual(posture(person, sit_ratio=1.8), "standing")
        self.assertEqual(posture(person, sit_ratio=2.5), "sitting")


class AssignmentTests(unittest.TestCase):
    def test_seated_person_on_chair(self):
        self.assertGreaterEqual(overlap_ratio(SEATED, CHAIRS[1]), 0.8)
        self.assertEqual(assign_people_to_chairs([SEATED], CHAIRS), {1: True, 2: False})

    def test_standing_person_in_front_of_chair_is_ignored(self):
        self.assertGreater(overlap_ratio(STANDING, CHAIRS[1]), 0.25)  # overlaps the chair...
        self.assertEqual(assign_people_to_chairs([STANDING], CHAIRS), {1: False, 2: False})  # ...but is standing

    def test_person_leaning_over_a_desk_next_to_a_chair_is_ignored(self):
        # wide enough to look "sitting" by shape, but far taller than the chair
        leaning = (60, 0, 260, 200)
        self.assertEqual(posture(leaning), "sitting")
        self.assertGreater(overlap_ratio(leaning, CHAIRS[1]), 0.9)
        self.assertEqual(assign_people_to_chairs([leaning], CHAIRS), {1: False, 2: False})
        # the limit can be relaxed for camera views where chairs look small
        self.assertEqual(assign_people_to_chairs([leaning], CHAIRS, max_height_ratio=4.0)[1], True)

    def test_matches_say_which_person_is_on_which_chair(self):
        people = [SEATED, STANDING, (190, 60, 270, 160)]
        self.assertEqual(match_people_to_chairs(people, CHAIRS), {0: 1, 2: 2})

    def test_far_person_is_ignored(self):
        far = (400, 60, 480, 160)
        self.assertEqual(assign_people_to_chairs([far], CHAIRS), {1: False, 2: False})

    def test_small_overlap_is_ignored(self):
        edge = (150, 100, 230, 120)  # sitting shape, only grazes the edge of chair 1
        self.assertLess(overlap_ratio(edge, CHAIRS[1]), 0.25)
        self.assertEqual(assign_people_to_chairs([edge], CHAIRS)[1], False)

    def test_person_only_counts_for_best_chair(self):
        wide = (90, 60, 215, 160)  # touches both chairs, covers chair 1 far more
        result = assign_people_to_chairs([wide], CHAIRS, min_overlap=0.1, sit_ratio=2.0)
        self.assertEqual(result, {1: True, 2: False})

    def test_two_people_two_chairs(self):
        people = [SEATED, (190, 60, 270, 160)]
        self.assertEqual(assign_people_to_chairs(people, CHAIRS), {1: True, 2: True})


class RealFrameRegression(unittest.TestCase):
    """Boxes read from a real test run: YOLO reported two chairs twice, and a person
    standing and leaning over a desk used to be mistaken for someone sitting down."""

    def test_real_frame(self):
        from chairs import suppress_nested

        chair_boxes = [
            (1035, 145, 1175, 340), (1055, 145, 1150, 238),   # one chair reported twice
            (330, 230, 485, 375), (305, 220, 420, 340),       # another chair reported twice
            (690, 365, 795, 490),                             # the chair the seated person uses
            (790, 160, 935, 382),                             # an empty chair
        ]
        chairs = dict(enumerate(suppress_nested(chair_boxes), start=1))
        self.assertEqual(len(chairs), 4)

        leaning_person = (245, 75, 510, 435)
        seated_person = (548, 296, 790, 510)
        occupied = assign_people_to_chairs([leaning_person, seated_person], chairs)
        self.assertEqual([cid for cid, taken in occupied.items() if taken], [3])
        self.assertEqual(chairs[3], (690, 365, 795, 490))


def run_timeline(tracker, seen_at, ids, end, step=0.5):
    t = 0.0
    while t <= end + 1e-9:
        tracker.update({i: seen_at(i, t) for i in ids}, t)
        t += step
    return end


class TrackerTests(unittest.TestCase):
    def test_sitting_is_timed(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        end = run_timeline(tracker, lambda i, t: 2.0 <= t < 8.0, [1], 12.0)
        tracker.finish(end)
        self.assertEqual(tracker.states[1].sessions, 1)
        self.assertAlmostEqual(tracker.states[1].total_seconds, 6.0, delta=0.01)

    def test_status_goes_through_the_stages(self):
        tracker = OccupancyTracker(occupy_after=2.0, vacate_after=2.0)
        tracker.update({1: False}, 0.0)
        self.assertEqual(tracker.states[1].status, "empty")
        tracker.update({1: True}, 1.0)
        self.assertEqual(tracker.states[1].status, "occupying")
        tracker.update({1: True}, 3.0)
        self.assertEqual(tracker.states[1].status, "occupied")
        tracker.update({1: False}, 4.0)
        self.assertEqual(tracker.states[1].status, "vacating")
        tracker.update({1: False}, 6.0)
        self.assertEqual(tracker.states[1].status, "empty")

    def test_brief_presence_is_ignored(self):
        tracker = OccupancyTracker(occupy_after=2.0, vacate_after=1.5)
        end = run_timeline(tracker, lambda i, t: 3.0 <= t < 4.5, [1], 10.0)
        tracker.finish(end)
        self.assertEqual(tracker.states[1].sessions, 0)

    def test_short_gap_does_not_split_the_sitting(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        end = run_timeline(tracker, lambda i, t: t < 10.0 and not (5.0 <= t < 5.5), [1], 12.0)
        tracker.finish(end)
        self.assertEqual(tracker.states[1].sessions, 1)
        self.assertAlmostEqual(tracker.states[1].total_seconds, 10.0, delta=0.01)

    def test_running_time_includes_current_sitting(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        run_timeline(tracker, lambda i, t: t >= 2.0, [1], 6.0)
        self.assertAlmostEqual(tracker.current_seconds(1, 6.0), 4.0, delta=0.01)

    def test_finish_closes_open_sitting(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        run_timeline(tracker, lambda i, t: t >= 2.0, [1], 6.0)
        tracker.finish(6.0)
        self.assertAlmostEqual(tracker.states[1].total_seconds, 4.0, delta=0.01)
        self.assertFalse(tracker.states[1].occupied)

    def test_chair_that_disappears_is_closed(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        for t in (0.0, 1.0, 2.0, 3.0):
            tracker.update({1: True}, t)
        for t in (4.0, 5.0, 6.0):  # chair 1 is no longer reported
            tracker.update({}, t)
        self.assertFalse(tracker.states[1].occupied)
        self.assertEqual(tracker.states[1].sessions, 1)

    def test_chairs_are_independent_and_sittings_listed(self):
        tracker = OccupancyTracker(occupy_after=1.0, vacate_after=1.5)
        end = run_timeline(tracker, lambda i, t: (1.0 <= t < 4.0) if i == 1 else (6.0 <= t < 9.0), [1, 2], 12.0)
        tracker.finish(end)
        self.assertEqual([(c, round(s), round(e)) for c, s, e in tracker.sittings()],
                         [(1, 1, 4), (2, 6, 9)])


if __name__ == "__main__":
    unittest.main()
