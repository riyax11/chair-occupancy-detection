"""Keeps track of chairs from frame to frame.

Chairs can be detected automatically (YOLO class 'chair') or marked by hand. Automatic
detections come and go and the camera may move, so detections are matched to existing
chair tracks. This gives every chair a stable id and keeps it alive through short
detection gaps.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from occupancy import Box, area, intersection_area, iou


def _center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _size(box: Box) -> float:
    return max(box[2] - box[0], box[3] - box[1], 1.0)


def containment(a: Box, b: Box) -> float:
    """How much of the smaller box lies inside the other box (0 to 1)."""
    smaller = min(area(a), area(b))
    return intersection_area(a, b) / smaller if smaller > 0 else 0.0


def suppress_nested(boxes: Sequence[Box], threshold: float = 0.7) -> List[Box]:
    """Drop boxes that mostly sit inside a larger box (YOLO often reports one chair twice)."""
    kept_indexes: List[int] = []
    for index in sorted(range(len(boxes)), key=lambda i: area(boxes[i]), reverse=True):
        if all(containment(boxes[index], boxes[other]) < threshold for other in kept_indexes):
            kept_indexes.append(index)
    return [tuple(boxes[i]) for i in sorted(kept_indexes)]  # type: ignore[misc]


@dataclass
class ChairTrack:
    id: int
    box: Box
    hits: int
    last_seen: float


class ChairTrackManager:
    """Tracks automatically detected chairs.

    - A new chair must be detected `min_hits` times before it is shown (ignores one-off ghosts).
    - A chair that is no longer detected is kept for `max_age` seconds at its last position.
    - `hold()` keeps a chair alive while a seated person is on it, because a chair is often
      hidden by the person who sits on it.
    """

    def __init__(self, max_age: float = 3.0, min_hits: int = 2, smooth: float = 0.6,
                 min_iou: float = 0.2, max_center_shift: float = 0.6, duplicate_threshold: float = 0.7):
        self.duplicate_threshold = duplicate_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.smooth = smooth
        self.min_iou = min_iou
        self.max_center_shift = max_center_shift
        self._tracks: List[ChairTrack] = []
        self._next_id = 1

    def update(self, detections: Sequence[Box], t: float) -> None:
        detections = suppress_nested(detections, self.duplicate_threshold)
        pairs = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                overlap = iou(track.box, det)
                (tx, ty), (dx, dy) = _center(track.box), _center(det)
                shift = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 / max(_size(track.box), _size(det))
                if overlap >= self.min_iou or shift <= self.max_center_shift:
                    pairs.append((shift - overlap, ti, di))  # lower is a better match
        pairs.sort()

        used_tracks, used_dets = set(), set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track, det = self._tracks[ti], detections[di]
            a = self.smooth
            track.box = tuple(a * n + (1 - a) * o for n, o in zip(det, track.box))  # type: ignore[assignment]
            track.hits += 1
            track.last_seen = t

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            if any(containment(det, tr.box) >= self.duplicate_threshold for tr in self._tracks):
                continue  # another view of a chair that is already tracked
            self._tracks.append(ChairTrack(self._next_id, tuple(det), 1, t))  # type: ignore[arg-type]
            self._next_id += 1

        self._merge_duplicate_tracks()
        self._tracks = [
            tr for tr in self._tracks
            if (t - tr.last_seen) <= (self.max_age if tr.hits >= self.min_hits else 1.0)
        ]

    def _merge_duplicate_tracks(self) -> None:
        """If two tracks cover the same chair, keep the one that has been seen more often."""
        tracks = sorted(self._tracks, key=lambda tr: (-tr.hits, tr.id))
        kept: List[ChairTrack] = []
        for track in tracks:
            if all(containment(track.box, other.box) < self.duplicate_threshold for other in kept):
                kept.append(track)
        self._tracks = sorted(kept, key=lambda tr: tr.id)

    def hold(self, chair_ids: Iterable[int], t: float) -> None:
        """Keep these chairs alive (a seated person is on them)."""
        wanted = set(chair_ids)
        for track in self._tracks:
            if track.id in wanted:
                track.last_seen = t

    def visible(self) -> Dict[int, Box]:
        return {tr.id: tr.box for tr in self._tracks if tr.hits >= self.min_hits}


class StaticChairs:
    """Hand-marked chairs (fixed camera). Same interface as ChairTrackManager."""

    def __init__(self, boxes: Sequence[Box]):
        self._chairs = {i + 1: tuple(b) for i, b in enumerate(boxes)}

    def update(self, detections: Sequence[Box], t: float) -> None:
        pass

    def hold(self, chair_ids: Iterable[int], t: float) -> None:
        pass

    def visible(self) -> Dict[int, Box]:
        return dict(self._chairs)
