"""Core occupancy logic.

This module needs neither OpenCV nor YOLO, so it is easy to test on its own.
Boxes are (x1, y1, x2, y2) tuples in pixel coordinates. Chairs are identified by
an integer id, so the same code works for hand-marked and automatically detected chairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    ox1, oy1 = max(a[0], b[0]), max(a[1], b[1])
    ox2, oy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def lower_body_box(person: Box, fraction: float = 0.5) -> Box:
    """The bottom part of a person box (default: the bottom half).

    A seated person's hips and legs are what overlap the chair.
    """
    x1, y1, x2, y2 = person
    return (x1, y2 - fraction * (y2 - y1), x2, y2)


def overlap_ratio(person: Box, chair: Box, lower_fraction: float = 0.5) -> float:
    """Share of the chair box that is covered by the person's lower body (0 to 1)."""
    chair_area = area(chair)
    if chair_area == 0:
        return 0.0
    return intersection_area(lower_body_box(person, lower_fraction), chair) / chair_area


# ----------------------------------------------------------------------------
# Posture
# ----------------------------------------------------------------------------
def posture(person: Box, sit_ratio: float = 1.8) -> str:
    """Guess 'sitting' or 'standing' from the shape of the person box.

    A standing person is tall and narrow (height / width is roughly 2.5 or more).
    A seated person is shorter and wider, especially when a desk hides the legs.
    This is a heuristic: it depends on the camera angle, so `sit_ratio` is tunable.
    """
    width = person[2] - person[0]
    height = person[3] - person[1]
    if width <= 0:
        return "standing"
    return "sitting" if (height / width) <= sit_ratio else "standing"


# ----------------------------------------------------------------------------
# Matching people to chairs (one frame)
# ----------------------------------------------------------------------------
def match_people_to_chairs(
    people: Sequence[Box],
    chairs: Mapping[int, Box],
    min_overlap: float = 0.25,
    sit_ratio: float = 1.8,
    lower_fraction: float = 0.5,
    max_height_ratio: float = 2.2,
) -> Dict[int, int]:
    """Decide which people are seated on which chair. Returns {person_index: chair_id}.

    A person counts as seated on a chair only if all of these hold:
    - their box is not tall and narrow (`sit_ratio`), so a walking person is ignored;
    - their box is not much taller than the chair (`max_height_ratio`), so someone who
      stands or leans over a desk next to a chair is ignored;
    - their lower body covers at least `min_overlap` of the chair box.
    Each person is assigned to at most one chair: the one they overlap most.
    """
    matches: Dict[int, int] = {}
    for person_index, person in enumerate(people):
        if posture(person, sit_ratio) != "sitting":
            continue
        person_height = person[3] - person[1]
        best_id: Optional[int] = None
        best_ratio = 0.0
        for chair_id, chair in chairs.items():
            chair_height = chair[3] - chair[1]
            if chair_height <= 0 or person_height / chair_height > max_height_ratio:
                continue
            ratio = overlap_ratio(person, chair, lower_fraction)
            if ratio > best_ratio:
                best_ratio, best_id = ratio, chair_id
        if best_id is not None and best_ratio >= min_overlap:
            matches[person_index] = best_id
    return matches


def assign_people_to_chairs(
    people: Sequence[Box],
    chairs: Mapping[int, Box],
    min_overlap: float = 0.25,
    sit_ratio: float = 1.8,
    lower_fraction: float = 0.5,
    max_height_ratio: float = 2.2,
) -> Dict[int, bool]:
    """For one frame, say which chairs have a seated person on them: {chair_id: bool}."""
    occupied = {chair_id: False for chair_id in chairs}
    matches = match_people_to_chairs(people, chairs, min_overlap, sit_ratio, lower_fraction, max_height_ratio)
    for chair_id in matches.values():
        occupied[chair_id] = True
    return occupied


# ----------------------------------------------------------------------------
# Stable states and timers (over time)
# ----------------------------------------------------------------------------
@dataclass
class ChairState:
    occupied: bool = False
    session_start: Optional[float] = None
    change_since: Optional[float] = None
    sittings: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(end - start for start, end in self.sittings)

    @property
    def sessions(self) -> int:
        return len(self.sittings)

    @property
    def status(self) -> str:
        """'empty', 'occupying' (person just sat down), 'occupied' or 'vacating'."""
        if self.change_since is not None:
            return "vacating" if self.occupied else "occupying"
        return "occupied" if self.occupied else "empty"

    def update(self, seen: bool, t: float, occupy_after: float, vacate_after: float) -> None:
        if seen == self.occupied:
            self.change_since = None  # nothing pending
            return
        if self.change_since is None:
            self.change_since = t
        wait = occupy_after if seen else vacate_after
        if t - self.change_since >= wait:
            if seen:
                self.occupied = True
                self.session_start = self.change_since
            else:
                self.occupied = False
                self.sittings.append((self.session_start or 0.0, max(self.change_since, self.session_start or 0.0)))
                self.session_start = None
            self.change_since = None

    def current_seconds(self, t: float) -> float:
        running = (t - self.session_start) if self.occupied and self.session_start is not None else 0.0
        return self.total_seconds + max(0.0, running)

    def finish(self, t: float) -> None:
        if self.occupied and self.session_start is not None:
            self.sittings.append((self.session_start, max(t, self.session_start)))
        self.occupied = False
        self.session_start = None
        self.change_since = None


class OccupancyTracker:
    """Turns noisy per-frame detections into stable occupied/empty states with timers.

    A chair becomes occupied only after a seated person has been on it for `occupy_after`
    seconds, and becomes empty only after `vacate_after` seconds without one. This stops
    flicker from missed detections and from people just passing by. Times are measured in
    video seconds, so results do not depend on how fast the computer processes the video.
    """

    def __init__(self, occupy_after: float = 2.0, vacate_after: float = 2.0):
        self.occupy_after = occupy_after
        self.vacate_after = vacate_after
        self.states: Dict[int, ChairState] = {}

    def update(self, seen: Mapping[int, bool], t: float) -> None:
        """Feed one frame's result: {chair_id: has_seated_person}. Chairs missing from
        `seen` (for example a chair that is no longer in view) count as not occupied."""
        for chair_id in seen:
            self.states.setdefault(chair_id, ChairState())
        for chair_id, state in self.states.items():
            state.update(bool(seen.get(chair_id, False)), t, self.occupy_after, self.vacate_after)

    def current_seconds(self, chair_id: int, t: float) -> float:
        state = self.states.get(chair_id)
        return state.current_seconds(t) if state else 0.0

    def finish(self, t: float) -> None:
        for state in self.states.values():
            state.finish(t)

    def sittings(self) -> List[Tuple[int, float, float]]:
        """All finished sittings as (chair_id, start_seconds, end_seconds)."""
        rows = [(cid, s, e) for cid, st in self.states.items() for s, e in st.sittings]
        return sorted(rows, key=lambda r: (r[1], r[0]))
