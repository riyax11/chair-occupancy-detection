import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chairs import ChairTrackManager, StaticChairs, suppress_nested  # noqa: E402


def shifted(box, dx, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


A = (100, 100, 160, 170)
B = (300, 120, 365, 190)


class TrackManagerTests(unittest.TestCase):
    def test_new_chair_needs_two_detections(self):
        m = ChairTrackManager(min_hits=2)
        m.update([A], 0.0)
        self.assertEqual(m.visible(), {})          # one-off detection is not shown yet
        m.update([A], 0.1)
        self.assertEqual(list(m.visible()), [1])

    def test_ids_stay_stable_when_the_camera_drifts(self):
        m = ChairTrackManager()
        for k in range(30):  # camera slowly pans 3 px per step (90 px in total)
            m.update([shifted(A, 3 * k), shifted(B, 3 * k)], k * 0.1)
        ids = sorted(m.visible())
        self.assertEqual(ids, [1, 2])
        # chair 1 is still the left-hand chair
        self.assertLess(m.visible()[1][0], m.visible()[2][0])

    def test_two_nearby_chairs_do_not_swap(self):
        m = ChairTrackManager()
        left, right = (100, 100, 160, 170), (175, 100, 235, 170)
        for k in range(10):
            m.update([left, right], k * 0.1)
        self.assertLess(m.visible()[1][0], m.visible()[2][0])

    def test_chair_survives_a_short_detection_gap(self):
        m = ChairTrackManager(max_age=3.0)
        m.update([A], 0.0)
        m.update([A], 0.1)
        m.update([], 1.5)
        self.assertEqual(list(m.visible()), [1])
        m.update([A], 2.0)  # comes back with the same id
        self.assertEqual(list(m.visible()), [1])

    def test_chair_is_dropped_after_max_age(self):
        m = ChairTrackManager(max_age=3.0)
        m.update([A], 0.0)
        m.update([A], 0.1)
        m.update([], 4.0)
        self.assertEqual(m.visible(), {})

    def test_hold_keeps_a_chair_covered_by_a_person(self):
        m = ChairTrackManager(max_age=3.0)
        m.update([A], 0.0)
        m.update([A], 0.1)
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            m.update([], t)       # the chair is hidden by the person sitting on it
            m.hold([1], t)
        self.assertEqual(list(m.visible()), [1])

    def test_one_off_ghost_disappears(self):
        m = ChairTrackManager(min_hits=2)
        m.update([A, B], 0.0)
        m.update([A], 0.1)        # B was a ghost
        m.update([A], 2.0)
        self.assertEqual(list(m.visible()), [1])


class DuplicateTests(unittest.TestCase):
    def test_nested_boxes_keep_only_the_larger_one(self):
        outer = (1035, 145, 1175, 340)
        inner = (1055, 145, 1150, 238)  # same chair reported twice
        self.assertEqual(suppress_nested([inner, outer]), [outer])

    def test_separate_chairs_are_both_kept(self):
        self.assertEqual(len(suppress_nested([A, B])), 2)

    def test_a_chair_reported_twice_gets_one_track(self):
        m = ChairTrackManager()
        outer, inner = (100, 100, 200, 260), (115, 110, 185, 190)
        for k in range(5):
            m.update([outer, inner], k * 0.1)
        self.assertEqual(list(m.visible()), [1])

    def test_a_late_duplicate_does_not_create_a_second_chair(self):
        m = ChairTrackManager()
        for k in range(4):
            m.update([A], k * 0.1)
        m.update([A, (110, 105, 150, 160)], 0.5)  # a part of chair A shows up as its own box
        m.update([A, (110, 105, 150, 160)], 0.6)
        self.assertEqual(list(m.visible()), [1])

    def test_existing_duplicate_tracks_are_merged(self):
        m = ChairTrackManager()
        m.update([A], 0.0)
        m.update([A], 0.1)
        # force a second overlapping track, as if it had been created earlier
        from chairs import ChairTrack
        m._tracks.append(ChairTrack(9, (110, 105, 150, 160), 5, 0.1))
        m._merge_duplicate_tracks()
        self.assertEqual(list(m.visible()), [9])  # the track seen more often is kept


class StaticChairsTests(unittest.TestCase):
    def test_hand_marked_chairs_get_ids_from_one(self):
        s = StaticChairs([A, B])
        s.update([(0, 0, 5, 5)], 1.0)  # detections are ignored
        self.assertEqual(s.visible(), {1: A, 2: B})


if __name__ == "__main__":
    unittest.main()
