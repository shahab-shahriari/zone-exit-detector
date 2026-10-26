"""Draw a zone polygon on the first frame of a video or stream and save it as normalized JSON."""
import json

import cv2


def draw_zone(source, out_path: str = "zone.json", max_width: int = 960) -> bool:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    ok, frame = False, None
    for _ in range(5):  # skip the first frames of a stream, they are often broken
        ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read a frame from the source")
        return False

    h, w = frame.shape[:2]
    scale = min(1.0, max_width / float(w))
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    pts = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    win = "draw zone"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("left click: add point | z: undo | c: clear | Enter: save | Esc: cancel")
    saved = False
    while True:
        view = disp.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 3, (0, 255, 255), -1)
            if i:
                cv2.line(view, pts[i - 1], p, (0, 255, 255), 2)
        if len(pts) >= 3:
            cv2.line(view, pts[-1], pts[0], (0, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key == ord("z") and pts:
            pts.pop()
        if key == ord("c"):
            pts.clear()
        if key == 13 and len(pts) >= 3:
            dh, dw = disp.shape[:2]
            norm = [[round(x / dw, 4), round(y / dh, 4)] for x, y in pts]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"points": norm}, f, indent=2)
            print(f"saved {out_path}")
            saved = True
            break
    cv2.destroyAllWindows()
    return saved
