from typing import List
from pathlib import Path
from autoplay.window import WindowObject
import win32gui
import pyautogui
import time
from pathlib import Path
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

def click_button(window, template, retry=True):
    # Sleep random amount of time so that we look slightly less like a bot
    # time.sleep(random.uniform(0.0, 3.0))

    left, top, right, bottom = win32gui.GetWindowRect(window.hwnd)
    # everything we want to click is in bottom half of the screen
    top = (top + bottom) // 2
    region = {"top": top, "left": left, "width": right - left, "height": bottom - top}
    frame = grab_region(region)

    boxes = find_template_all(frame, template, threshold=0.90)
    # logger.debug(boxes)
    
    # Tenhou event is faster than game UI update. The tile might not be visible yet
    # Sleep and retries once to ensure we don't miss this case
    # this should also cover the dora affecting recognition case
    i = 0
    while not boxes and retry and i < 3:
        time.sleep(2)
        frame = grab_region(region)
        boxes = find_template_all(frame, template, threshold=0.80 - i * 0.1)
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

def resize_template(img, scale=0.7):
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def merge_templates_horiz(img_a, img_b, gap=0):
    h = max(img_a.shape[0], img_b.shape[0])

    def pad_to_h(img):
        pad_bottom = h - img.shape[0]
        return cv2.copyMakeBorder(
            img, 0, pad_bottom, 0, 0,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

    img_a = pad_to_h(img_a)
    img_b = pad_to_h(img_b)

    parts = [img_a]
    if gap > 0:
        spacer = np.zeros((h, gap, img_a.shape[2]), dtype=img_a.dtype)
        parts.append(spacer)
    parts.append(img_b)

    return np.hstack(parts)

def merge_chi_pairs(template1_path, template2_path):
    template1 = cv2.imread(str(template1_path), cv2.IMREAD_COLOR)
    template2 = cv2.imread(str(template2_path), cv2.IMREAD_COLOR)
    template1_small = resize_template(template1, 0.8)
    template2_small = resize_template(template2, 0.8)
    merged = merge_templates_horiz(template1_small, template2_small, gap=2)
    return merged

def find_template_all(frame, template, threshold=0.88):
    """
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    coordinates are relative to frame
    """
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)

    points = list(zip(xs, ys))
    points = group_points(points, distance=10)

    h, w = template.shape[:2]
    boxes = [(x, y, x + w, y + h) for x, y in points]
    return boxes

class MajsoulAutoPlay():
    def __init__(self):
        pass

    def check_window(self, window):
        if "雀魂" in window.name or "Mahjong Soul" in window.name:
            return True
        else:
            return False

    def auto_select_window(self, windows: List[WindowObject]):
        for window in windows:
            if "雀魂" in window.name or "Mahjong Soul" in window.name:
                logger.info(window.name)
                return window
        return None

    def click_discard(self, window: WindowObject, mjai_msg):
        pai = mjai_msg["pai"]
        if pai not in VALID_PAI:
            return False
        try:
            template_path = BASE_DIR / "assets" / "pais" / f"{pai}.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            return click_button(window, template)
        except Exception as e:
            logger.error(f"Failed to click discard {pai}: ", e)
            return False
    
    def click_action(self, window: WindowObject, mjai_msg):
        action = mjai_msg["type"]
        if action not in VALID_ACTIONS:
            return False
        try:
            if action in ["daiminkan", "kakan", "ankan"]:
                action = "kan"
            template_path = BASE_DIR / "assets" / "actions" / f"{action}.png"
            if action == "hora" and mjai_msg["actor"] == mjai_msg["target"]:
                # tsumo is a different button...
                template_path = BASE_DIR / "assets" / "actions" / "hora2.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            success = click_button(window, template)
            
            if success and action == "chi":
                consumed = mjai_msg["consumed"]
                template_path1 = BASE_DIR / "assets" / "pais" / f"{consumed[0]}.png"
                template_path2 = BASE_DIR / "assets" / "pais" / f"{consumed[1]}.png"
                click_button(window, merge_chi_pairs(template_path1, template_path2), retry=False)
            if success and action == "reach":
                self.click_discard(window, mjai_msg)
            return success
        except Exception as e:
            logger.error(f"Failed to click action {action}: ", e)
            return False

