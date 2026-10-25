"""End-to-end tests of the video loop using synthetic videos and fake detectors.

No YOLO model is needed: the detector is replaced by a function that returns scripted
people and chairs, so the whole loop (timing, tracking, drawing, reports) is exercised.
"""

import csv
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Detections, anonymize, load_chairs, run  # noqa: E402

FPS = 10
TOTAL_FRAMES = 150  # 15 seconds
CHAIR_1 = (100, 100, 160, 170)
CHAIR_2 = (220, 100, 280, 170)
SEATED_ON_1 = (90, 60, 170, 170)
WALKING = (40, 20, 80, 180)  # tall and narrow


def make_video(path, frames=TOTAL_FRAMES):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    if not writer.isOpened():
        return False
    for _ in range(frames):
        writer.write(np.full((240, 320, 3), 60, dtype=np.uint8))
    writer.release()
    return True


class PipelineTests(unittest.TestCase):
    def test_automatic_chairs_with_a_drifting_camera(self):
        """A person sits on chair 1 from 3.0 s to 11.9 s while the camera drifts sideways."""
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "synthetic.mp4")
            if not make_video(video):
                self.skipTest("this OpenCV build cannot write mp4 files")

            counter = {"n": -1}

            def shift(box, dx):
                return (box[0] + dx, box[1], box[2] + dx, box[3])

            def detector(frame):
                counter["n"] += 1
                n = counter["n"]
                dx = n // 5  # the whole scene slides 1 px every 5 frames
                chairs = [shift(CHAIR_1, dx), shift(CHAIR_2, dx)]
                people = [shift(WALKING, dx)]  # someone walking about all the time
                if 30 <= n < 120:
                    people.append(shift(SEATED_ON_1, dx))
                return Detections(people, chairs)

            report = os.path.join(tmp, "report.csv")
            sittings = os.path.join(tmp, "sittings.csv")
            output = os.path.join(tmp, "annotated.mp4")
            tracker = run(video, detector, stride=1, show=False,
                          report=report, sittings=sittings, output=output)

            self.assertEqual(sorted(tracker.states), [1, 2])
            self.assertEqual(tracker.states[1].sessions, 1)
            self.assertAlmostEqual(tracker.states[1].total_seconds, 9.0, delta=0.2)
            self.assertEqual(tracker.states[2].sessions, 0)  # the walker never "occupies" a chair

            with open(report) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["sittings"] for r in rows], ["1", "0"])
            with open(sittings) as f:
                sit_rows = list(csv.DictReader(f))
            self.assertEqual(len(sit_rows), 1)
            self.assertAlmostEqual(float(sit_rows[0]["start_s"]), 3.0, delta=0.2)
            self.assertTrue(os.path.getsize(output) > 0)

    def test_duplicate_chair_boxes_and_a_leaning_person_do_not_cause_false_occupancy(self):
        """YOLO reports one chair twice; a person leans over a desk beside it for 8 seconds."""
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "synthetic.mp4")
            if not make_video(video):
                self.skipTest("this OpenCV build cannot write mp4 files")
            leaning = (60, 0, 260, 200)  # wide box, far taller than the chair

            def detector(frame):
                inner = (CHAIR_1[0] + 10, CHAIR_1[1] + 5, CHAIR_1[2] - 10, CHAIR_1[3] - 25)
                return Detections([leaning], [CHAIR_1, inner])

            tracker = run(video, detector, stride=1, show=False)
            self.assertEqual(sorted(tracker.states), [1])          # one chair, not two
            self.assertEqual(tracker.states[1].sessions, 0)        # and nobody is sitting on it

    def test_hand_marked_chairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "synthetic.mp4")
            if not make_video(video):
                self.skipTest("this OpenCV build cannot write mp4 files")
            counter = {"n": -1}

            def detector(frame):
                counter["n"] += 1
                people = [SEATED_ON_1] if 20 <= counter["n"] < 100 else []
                return Detections(people, [])  # chair detections are not used in this mode

            tracker = run(video, detector, chairs=[CHAIR_1, CHAIR_2], stride=1, show=False)
            self.assertAlmostEqual(tracker.states[1].total_seconds, 8.0, delta=0.2)
            self.assertEqual(tracker.states[2].total_seconds, 0.0)

    def test_stride_skips_frames_but_keeps_video_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "synthetic.mp4")
            if not make_video(video):
                self.skipTest("this OpenCV build cannot write mp4 files")

            def detector(frame):
                return Detections([SEATED_ON_1], [])

            tracker = run(video, detector, chairs=[CHAIR_1], stride=5, show=False)
            self.assertAlmostEqual(tracker.states[1].total_seconds, 15.0, delta=0.6)

    def test_file_without_video_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "not_a_video.mp4")
            with open(fake, "wb") as f:
                f.write(b"this is not a video file")
            with self.assertRaises(RuntimeError):
                run(fake, lambda frame: Detections([], []), show=False)

    def test_missing_chairs_file_gives_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_chairs("does_not_exist.json")


class AnonymizeTests(unittest.TestCase):
    def test_head_area_is_blurred_and_the_rest_is_untouched(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        original = frame.copy()
        anonymize(frame, [(50, 20, 130, 180)])
        head_before = original[20:48, 50:130]
        head_after = frame[20:48, 50:130]
        self.assertFalse(np.array_equal(head_before, head_after))
        self.assertLess(head_after.std(), head_before.std())          # smoother
        self.assertTrue(np.array_equal(original[100:180, 50:130], frame[100:180, 50:130]))  # body untouched

    def test_boxes_outside_the_frame_do_not_crash(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        anonymize(frame, [(-50, -50, 20, 20), (90, 90, 300, 300), (10, 10, 10, 10)])


if __name__ == "__main__":
    unittest.main()
