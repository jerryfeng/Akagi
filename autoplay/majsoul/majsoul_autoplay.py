from typing import List, Optional, Dict, Tuple
from pathlib import Path
from autoplay.window import WindowObject
import win32gui
import pyautogui
import time
import cv2
import numpy as np
import mss
from autoplay.logger import logger
import random

BASE_DIR = Path(__file__).resolve().parent
VALID_PAI = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr"
]
VALID_ACTIONS = [
    "chi",
    "pon",
    "daiminkan",
    "kakan",
    "ankan",
    "none",
    "reach",
    "hora"
]


def grab_region(region):
    """
    region = dict(top=..., left=..., width=..., height=...)
    returns BGR image
    """
    with mss.mss() as sct:
        shot = np.array(sct.grab(region))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def group_points(points, distance=12):
    """
    Deduplicate nearby match points.
    points: list of (x, y)
    """
    groups = []
    for x, y in points:
        placed = False
        for g in groups:
            gx, gy, count = g
            if abs(x - gx) <= distance and abs(y - gy) <= distance:
                new_count = count + 1
                g[0] = (gx * count + x) / new_count
                g[1] = (gy * count + y) / new_count
                g[2] = new_count
                placed = True
                break
        if not placed:
            groups.append([float(x), float(y), 1])
    return [(int(gx), int(gy)) for gx, gy, _ in groups]


def click_button(window, template, retry=True, isHaku=False):
    if isHaku:
        time.sleep(2)

    left, top, right, bottom = win32gui.GetWindowRect(window.hwnd)
    top = (top + bottom) // 2
    region = {"top": top, "left": left, "width": right - left, "height": bottom - top}
    frame = grab_region(region)

    boxes = find_template_all(frame, template, threshold=0.90, isHaku=isHaku)

    i = 0
    while not boxes and retry and i < 3:
        time.sleep(2)
        frame = grab_region(region)
        boxes = find_template_all(frame, template, threshold=0.80 - i * 0.1, isHaku=isHaku)
        i += 1

    if boxes:
        x1, y1, x2, y2 = boxes[0]
        x1 += left
        y1 += top
        x2 += left
        y2 += top
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        pyautogui.moveTo(xc, yc, duration=0.5)
        time.sleep(0.5)
        pyautogui.click()
        time.sleep(0.5)
        xmid = (left + right) / 2
        pyautogui.moveTo(xmid, top, duration=0.5)
        return True
    else:
        return False


def resize_template(img, scale=0.75):
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def find_template_all(frame, template, threshold=0.88, isHaku=False):
    """
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    coordinates are relative to frame
    """
    templateMatchMode = cv2.TM_SQDIFF_NORMED if isHaku else cv2.TM_CCOEFF_NORMED

    result = cv2.matchTemplate(frame, template, templateMatchMode)
    ys, xs = np.where(result <= 0.08) if isHaku else np.where(result >= threshold)

    points = list(zip(xs, ys))
    points = group_points(points, distance=10)

    h, w = template.shape[:2]
    boxes = [(x, y, x + w, y + h) for x, y in points]
    return boxes


# ------------------------------------------------------------------
# Chi pair detection helpers
# ------------------------------------------------------------------

def get_window_rect(window: WindowObject) -> Tuple[int, int, int, int]:
    return win32gui.GetWindowRect(window.hwnd)


def get_chi_panel_region(window: WindowObject) -> Dict[str, int]:
    """
    Crop only the central top panel where Mahjong Soul shows the three chi options.
    Using a focused region makes pair detection far more stable than searching the whole screen.
    """
    left, top, right, bottom = get_window_rect(window)
    width = right - left
    height = bottom - top

    region = {
        "left": left + int(width * 0.20),
        "top": top + int(height * 0.6),
        "width": int(width * 0.80),
        "height": int(height * 0.25),
    }
    return region



def load_tile_template(pai: str):
    template_path = BASE_DIR / "assets" / "pais" / f"{pai}.png"
    return cv2.imread(str(template_path), cv2.IMREAD_COLOR)



def match_template_boxes_scored(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.84,
    scales: Tuple[float, ...] = (0.68, 0.72, 0.76, 0.80),
    distance: int = 14,
    is_haku: bool = False,
) -> List[Dict[str, float]]:
    """
    Returns scored detections for one tile template across a few scale hypotheses.
    Output items contain x1, y1, x2, y2, cx, cy, score, w, h.
    """
    if template is None:
        return []

    mode = cv2.TM_SQDIFF_NORMED if is_haku else cv2.TM_CCOEFF_NORMED
    raw_matches: List[Dict[str, float]] = []

    for scale in scales:
        tpl = resize_template(template, scale)
        th, tw = tpl.shape[:2]
        if th <= 0 or tw <= 0:
            continue
        if th > frame.shape[0] or tw > frame.shape[1]:
            continue

        result = cv2.matchTemplate(frame, tpl, mode)
        ys, xs = np.where(result <= 0.08) if is_haku else np.where(result >= threshold)

        for x, y in zip(xs, ys):
            score = float(1.0 - result[y, x]) if is_haku else float(result[y, x])
            raw_matches.append({
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + tw),
                "y2": int(y + th),
                "cx": float(x + tw / 2),
                "cy": float(y + th / 2),
                "score": score,
                "w": int(tw),
                "h": int(th),
            })

    if not raw_matches:
        return []

    # deduplicate nearby matches while keeping the highest-scoring one
    raw_matches.sort(key=lambda m: m["score"], reverse=True)
    deduped: List[Dict[str, float]] = []
    for m in raw_matches:
        if any(abs(m["cx"] - d["cx"]) <= distance and abs(m["cy"] - d["cy"]) <= distance for d in deduped):
            continue
        deduped.append(m)

    return deduped



def choose_best_chi_pair(
    detections_a: List[Dict[str, float]],
    detections_b: List[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """
    Build the best left-right tile pair from two individual tile detections.
    The popup options are arranged horizontally, so the correct pair should:
      - share nearly the same y
      - have the second tile to the right of the first
      - have a reasonable horizontal gap
    """
    best = None
    best_score = -1e18

    for a in detections_a:
        for b in detections_b:
            # prevent same physical box being reused when tiles are equal, e.g. 4s 4s
            if abs(a["cx"] - b["cx"]) < max(6, 0.20 * min(a["w"], b["w"])) and abs(a["cy"] - b["cy"]) < max(6, 0.20 * min(a["h"], b["h"])):
                continue

            if b["cx"] <= a["cx"]:
                continue

            y_penalty = abs(a["cy"] - b["cy"])
            if y_penalty > 18:
                continue

            gap = b["x1"] - a["x2"]
            avg_w = 0.5 * (a["w"] + b["w"])
            # Popup tiles are close to each other, usually with a small positive gap.
            if gap < -6 or gap > avg_w * 0.65:
                continue

            gap_penalty = abs(gap - avg_w * 0.08)
            score = (
                a["score"] + b["score"]
                - 0.030 * y_penalty
                - 0.015 * gap_penalty
            )

            candidate = {
                "x1": min(a["x1"], b["x1"]),
                "y1": min(a["y1"], b["y1"]),
                "x2": max(a["x2"], b["x2"]),
                "y2": max(a["y2"], b["y2"]),
                "score": score,
            }
            if score > best_score:
                best_score = score
                best = candidate

    return best



def click_chi_pair(window: WindowObject, consumed: List[str], retries: int = 4) -> bool:
    """
    Click the specific chi option by locating the two consumed tiles directly,
    instead of relying on one merged template.
    This is more robust because the popup tiles are scaled and spaced differently
    from the player's hand tiles.
    """
    if len(consumed) != 2:
        return False

    template_a = load_tile_template(consumed[0])
    template_b = load_tile_template(consumed[1])
    if template_a is None or template_b is None:
        logger.error(f"Missing chi tile templates for {consumed}")
        return False

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(0.8)

        region = get_chi_panel_region(window)
        frame = grab_region(region)

        threshold = max(0.76, 0.84 - 0.02 * attempt)
        det_a = match_template_boxes_scored(
            frame,
            template_a,
            threshold=threshold,
            is_haku=(consumed[0] == "P"),
        )
        det_b = match_template_boxes_scored(
            frame,
            template_b,
            threshold=threshold,
            is_haku=(consumed[1] == "P"),
        )

        logger.debug(f"chi detect {consumed} attempt={attempt} matches_a={len(det_a)} matches_b={len(det_b)}")
        pair_box = choose_best_chi_pair(det_a, det_b)
        if pair_box is None:
            continue

        x1 = region["left"] + int(pair_box["x1"])
        y1 = region["top"] + int(pair_box["y1"])
        x2 = region["left"] + int(pair_box["x2"])
        y2 = region["top"] + int(pair_box["y2"])
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2

        pyautogui.moveTo(xc, yc, duration=0.5)
        time.sleep(0.5)
        pyautogui.click()
        time.sleep(0.5)
        return True

class MajsoulAutoPlay():
    def __init__(self):
        pass

    def check_window(self, window):
        if "雀魂" in window.name or "Mahjong Soul" in window.name:
            return True
        else:
            return False

    def start_game(self):
        time.sleep(8)

    def end_game(self, window: WindowObject):
        try:
            time.sleep(25)
            template_path = BASE_DIR / "assets" / "actions" / "ok.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)
            time.sleep(3)
            pyautogui.click()
            time.sleep(3)
            template_path = BASE_DIR / "assets" / "actions" / "ok2.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)
            time.sleep(3)
            pyautogui.click()
            time.sleep(3)
            template_path = BASE_DIR / "assets" / "actions" / "play_again.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)
            time.sleep(2)
            template_path = BASE_DIR / "assets" / "actions" / "play_again_ok.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)

            # retry once to make sure we clicked
            time.sleep(3)
            template_path = BASE_DIR / "assets" / "actions" / "play_again.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)
            time.sleep(2)
            template_path = BASE_DIR / "assets" / "actions" / "play_again_ok.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            click_button(window, template)
            
            return True
        except Exception as e:
            logger.error(f"Failed to start another game: {e}")
            return False

    def auto_select_window(self, windows: List[WindowObject]):
        for window in windows:
            if "雀魂" in window.name or "Mahjong Soul" in window.name:
                left, top, right, bottom = win32gui.GetWindowRect(window.hwnd)
                logger.info(window.name)
                logger.info(f"Window width: {right-left}, height: {bottom-top}")
                return window
        return None

    def click_discard(self, window: WindowObject, mjai_msg):
        pai = mjai_msg["pai"]
        time.sleep(1)
        if pai not in VALID_PAI:
            return False
        try:
            template_path = BASE_DIR / "assets" / "pais" / f"{pai}.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            isHaku = True if pai == "P" else False
            return click_button(window, template, isHaku=isHaku)
        except Exception as e:
            logger.error(f"Failed to click discard {pai}: {e}")
            return False

    def click_action(self, window: WindowObject, mjai_msg):
        action = mjai_msg["type"]
        time.sleep(2)
        if action not in VALID_ACTIONS:
            return False
        try:
            if action in ["daiminkan", "kakan", "ankan"]:
                action = "kan"
            template_path = BASE_DIR / "assets" / "actions" / f"{action}.png"
            if action == "hora" and mjai_msg["actor"] == mjai_msg["target"]:
                template_path = BASE_DIR / "assets" / "actions" / "hora2.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            success = click_button(window, template)

            if action != "none":
                time.sleep(2)

            if success and action == "chi":
                consumed = mjai_msg["consumed"]
                if mjai_msg["chi_count"] > 1:
                    success = click_chi_pair(window, consumed)
            if success and action == "reach":
                self.click_discard(window, mjai_msg)
            return success
        except Exception as e:
            logger.error(f"Failed to click action {action}: {e}")
            return False
