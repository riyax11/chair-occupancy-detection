"""Optional: mark chairs by hand on the first frame of a video and save them to chairs.json.

Only needed for a fixed camera. By default main.py detects chairs automatically.

Click and drag to draw a box around each chair.
Keys:  u = undo last box   s = save and quit   q or Esc = quit without saving
"""

import argparse
import json
import sys

import cv2


def main() -> int:
    p = argparse.ArgumentParser(description="Mark chair boxes on a video frame.")
    p.add_argument("--video", required=True, help="video file path, or 0 for the webcam")
    p.add_argument("--output", default="chairs.json", help="where to save the boxes")
    p.add_argument("--frame", type=int, default=0, help="which frame number to use")
    args = p.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open '{args.video}'", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Error: no video frame could be read. Check that the file really contains video "
              "(an audio-only file renamed to .mp4 will not work).", file=sys.stderr)
        return 1

    chairs = []
    drag_start = None
    drag_now = None

    def on_mouse(event, x, y, flags, param):
        nonlocal drag_start, drag_now
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start, drag_now = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1, x2 = sorted((drag_start[0], x))
            y1, y2 = sorted((drag_start[1], y))
            if x2 - x1 > 5 and y2 - y1 > 5:
                chairs.append([x1, y1, x2, y2])
                print(f"Chair {len(chairs)}: {[x1, y1, x2, y2]}")
            drag_start, drag_now = None, None

    window = "Mark chairs (drag = box, u = undo, s = save, q = quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    saved = False
    while True:
        canvas = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(chairs, start=1):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(canvas, f"Chair {i}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if drag_start is not None and drag_now is not None:
            cv2.rectangle(canvas, drag_start, drag_now, (0, 255, 255), 1)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and chairs:
            chairs.pop()
        elif key == ord("s"):
            saved = True
            break
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    if saved and chairs:
        with open(args.output, "w") as f:
            json.dump(chairs, f, indent=2)
        print(f"Saved {len(chairs)} chair(s) to {args.output}")
    elif saved:
        print("No chairs were marked, nothing saved.")
    else:
        print("Quit without saving.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
