"""Chair occupancy detection with YOLOv8 and OpenCV.

Finds chairs and people in a video, decides which chairs have a seated person on them,
and times how long each chair is used.

Examples:
    python main.py --video demo.mp4 --report report.csv --output annotated.mp4
    python main.py --video demo.mp4 --chairs chairs.json     # hand-marked chairs, fixed camera
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2

from chairs import ChairTrackManager, StaticChairs
from occupancy import Box, OccupancyTracker, match_people_to_chairs

# Colours are BGR
EMPTY_COLOR = (0, 200, 0)
PENDING_COLOR = (0, 180, 255)
OCCUPIED_COLOR = (0, 0, 255)
SITTING_COLOR = (255, 160, 0)
STANDING_COLOR = (200, 200, 200)

COCO_PERSON = 0
COCO_CHAIR = 56


class Detections(NamedTuple):
    people: List[Box]
    chairs: List[Box]


Detector = Callable[[object], Detections]


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def make_yolo_detector(model_path: str, person_conf: float, chair_conf: float) -> Detector:
    """Return a function that finds people and chairs in a frame with one YOLO pass."""
    from ultralytics import YOLO  # imported here so the rest works without it

    model = YOLO(model_path)
    lowest = min(person_conf, chair_conf)

    def detect(frame) -> Detections:
        people: List[Box] = []
        chairs: List[Box] = []
        for result in model(frame, conf=lowest, classes=[COCO_PERSON, COCO_CHAIR], verbose=False):
            for box in result.boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                coords = tuple(box.xyxy[0].tolist())
                if cls == COCO_PERSON and score >= person_conf:
                    people.append(coords)
                elif cls == COCO_CHAIR and score >= chair_conf:
                    chairs.append(coords)
        return Detections(people, chairs)

    return detect


def load_chairs(path: Union[str, Path]) -> List[Tuple[int, int, int, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chair file '{path}' not found. Run select_chairs.py first, or leave out "
            "--chairs to detect chairs automatically."
        )
    chairs = []
    for item in json.loads(path.read_text()):
        if len(item) != 4:
            raise ValueError(f"Each chair needs 4 numbers [x1, y1, x2, y2], got: {item}")
        chairs.append(tuple(int(v) for v in item))
    if not chairs:
        raise ValueError(f"'{path}' contains no chairs.")
    return chairs


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def anonymize(frame, people: Sequence[Box], head_fraction: float = 0.35) -> None:
    """Blur the top part (the head) of every person box, in place."""
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in people:
        top, bottom = int(max(0, y1)), int(min(height, y1 + head_fraction * (y2 - y1)))
        left, right = int(max(0, x1)), int(min(width, x2))
        if bottom - top < 2 or right - left < 2:
            continue
        region = frame[top:bottom, left:right]
        k = max(3, (min(region.shape[:2]) // 2) | 1)  # odd kernel size
        frame[top:bottom, left:right] = cv2.GaussianBlur(region, (k, k), 0)


def draw_frame(frame, chairs: Dict[int, Box], people: Sequence[Box], seated_on: Dict[int, int],
               tracker: OccupancyTracker, t: float) -> None:
    for person_index, person in enumerate(people):
        x1, y1, x2, y2 = (int(v) for v in person)
        chair_id = seated_on.get(person_index)
        color = SITTING_COLOR if chair_id is not None else STANDING_COLOR
        label = f"Seated (chair {chair_id})" if chair_id is not None else "Person"
        label_y = y1 - 6 if y1 >= 40 else y1 + 20  # keep the label clear of the top bar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for chair_id, box in chairs.items():
        x1, y1, x2, y2 = (int(v) for v in box)
        state = tracker.states.get(chair_id)
        status = state.status if state else "empty"
        color = {"occupied": OCCUPIED_COLOR, "vacating": OCCUPIED_COLOR,
                 "occupying": PENDING_COLOR}.get(status, EMPTY_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if status == "occupied" else 2)
        label = {"occupied": "OCCUPIED", "vacating": "OCCUPIED", "occupying": "SITTING DOWN...",
                 "empty": "EMPTY"}[status]
        seconds = tracker.current_seconds(chair_id, t)
        text = f"Chair {chair_id}: {label}" + (f" {seconds:.0f}s" if seconds >= 1 else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y2 + th + 6
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    occupied_now = sum(1 for cid in chairs if tracker.states.get(cid) and tracker.states[cid].occupied)
    hud = f"Occupied chairs: {occupied_now}/{len(chairs)}    t = {t:.1f}s"
    (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (4, 4), (hw + 16, hh + 20), (0, 0, 0), -1)
    cv2.putText(frame, hud, (10, hh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(
    video: Union[str, int],
    detect: Detector,
    chairs: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    stride: int = 3,
    min_overlap: float = 0.25,
    sit_ratio: float = 1.8,
    max_height_ratio: float = 2.2,
    occupy_after: float = 2.0,
    vacate_after: float = 2.0,
    chair_max_age: float = 3.0,
    output: Optional[str] = None,
    report: Optional[str] = None,
    sittings: Optional[str] = None,
    blur_people: bool = False,
    show: bool = True,
) -> OccupancyTracker:
    """Process a video (or webcam index) and return the tracker holding the final results.

    If `chairs` is given, those hand-marked boxes are used. Otherwise chairs are detected
    automatically and tracked, which also works when the camera moves a little.
    """
    is_camera = isinstance(video, int)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {video}. Check the path, and that the file really "
            "contains video (an audio-only file renamed to .mp4 will not work)."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not is_camera and fps <= 0:
        fps = 30.0  # fall back when the file does not report a frame rate

    chair_source = StaticChairs(chairs) if chairs else ChairTrackManager(max_age=chair_max_age)
    tracker = OccupancyTracker(occupy_after, vacate_after)
    writer = None
    people: Sequence[Box] = []
    seated_on: Dict[int, int] = {}
    frame_index = 0
    t = 0.0
    start_clock = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = (time.time() - start_clock) if is_camera else frame_index / fps

            if frame_index % max(1, stride) == 0:
                found = detect(frame)
                people = found.people
                chair_source.update(found.chairs, t)
                visible = chair_source.visible()
                seated_on = match_people_to_chairs(people, visible, min_overlap, sit_ratio,
                                                   max_height_ratio=max_height_ratio)
                seen = {chair_id: False for chair_id in visible}
                for chair_id in seated_on.values():
                    seen[chair_id] = True
                chair_source.hold(list(seated_on.values()), t)
                tracker.update(seen, t)

            if blur_people:
                anonymize(frame, people)
            draw_frame(frame, chair_source.visible(), people, seated_on, tracker, t)

            if output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps if fps > 0 else 20.0, (w, h))
                writer.write(frame)

            if show:
                cv2.imshow("Chair Occupancy (press q or Esc to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError(
            f"No video frames could be read from '{video}'. Check the path, and that the "
            "file really contains video (an audio-only file renamed to .mp4 will not work)."
        )

    tracker.finish(t)
    print_summary(tracker, t)
    if report:
        write_report(tracker, report)
    if sittings:
        write_sittings(tracker, sittings)
    return tracker


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def print_summary(tracker: OccupancyTracker, duration: float) -> None:
    print(f"\nProcessed {duration:.1f} seconds of video")
    if not tracker.states:
        print("  No chairs were tracked.")
    for chair_id, state in sorted(tracker.states.items()):
        print(f"  Chair {chair_id}: occupied {state.total_seconds:.1f}s in {state.sessions} sitting(s)")


def write_report(tracker: OccupancyTracker, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chair", "occupied_seconds", "sittings"])
        for chair_id, state in sorted(tracker.states.items()):
            writer.writerow([chair_id, round(state.total_seconds, 1), state.sessions])
    print(f"Report saved to {path}")


def write_sittings(tracker: OccupancyTracker, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chair", "start_s", "end_s", "duration_s"])
        for chair_id, start, end in tracker.sittings():
            writer.writerow([chair_id, round(start, 1), round(end, 1), round(end - start, 1)])
    print(f"Sittings saved to {path}")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detect which chairs are occupied in a video.")
    p.add_argument("--video", required=True, help="video file path, or 0 for the webcam")
    p.add_argument("--chairs", help="optional chairs.json from select_chairs.py (fixed camera); "
                                    "leave out to detect chairs automatically")
    p.add_argument("--model", default="yolov8s.pt", help="YOLO weights (downloaded automatically)")
    p.add_argument("--conf", type=float, default=0.4, help="person detection confidence")
    p.add_argument("--chair-conf", type=float, default=0.25, help="chair detection confidence")
    p.add_argument("--stride", type=int, default=3, help="run detection on every Nth frame")
    p.add_argument("--min-overlap", type=float, default=0.25,
                   help="share of the chair box a seated person must cover (0 to 1)")
    p.add_argument("--sit-ratio", type=float, default=1.8,
                   help="a person box taller than this (height / width) counts as standing")
    p.add_argument("--max-height-ratio", type=float, default=2.2,
                   help="a person box more than this many times taller than the chair box is not seated on it")
    p.add_argument("--occupy-after", type=float, default=2.0,
                   help="seconds a person must sit on a chair before it counts as occupied")
    p.add_argument("--vacate-after", type=float, default=2.0,
                   help="seconds without a seated person before a chair counts as empty")
    p.add_argument("--chair-max-age", type=float, default=3.0,
                   help="seconds to keep a chair that is no longer detected")
    p.add_argument("--output", help="save the annotated video (e.g. annotated.mp4)")
    p.add_argument("--report", help="save per-chair totals as CSV")
    p.add_argument("--sittings", help="save every sitting (start, end, duration) as CSV")
    p.add_argument("--anonymize", action="store_true", help="blur people's heads in the output")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    video: Union[str, int] = int(args.video) if args.video.isdigit() else args.video
    try:
        chairs = load_chairs(args.chairs) if args.chairs else None
        detect = make_yolo_detector(args.model, args.conf, args.chair_conf)
        run(video, detect, chairs=chairs, stride=args.stride, min_overlap=args.min_overlap,
            sit_ratio=args.sit_ratio, max_height_ratio=args.max_height_ratio,
            occupy_after=args.occupy_after, vacate_after=args.vacate_after,
            chair_max_age=args.chair_max_age, output=args.output, report=args.report,
            sittings=args.sittings, blur_people=args.anonymize, show=not args.no_display)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print("Error: the 'ultralytics' package is missing. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
