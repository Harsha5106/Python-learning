"""
keep_alive_move.py - every 2 minutes, wiggles the mouse and/or switches tab.

Setup:   pip install pyautogui
Run:     python keep_alive_move.py
Stop:    Ctrl+C in the terminal (or move the mouse into a screen corner,
         which triggers pyautogui's built-in fail-safe).

Modes (set MODE below):
  "mouse" - nudge the mouse a few pixels and back
  "tab"   - press Ctrl+Tab to go to the next browser tab
  "both"  - do both each cycle
"""

import platform
import random
import time
from datetime import datetime

import pyautogui

INTERVAL_SECONDS = 5   # 2 minutes
MODE = "both"            # "mouse", "tab", or "both"
WIGGLE_PIXELS = 40       # how far the mouse moves

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.2

IS_MAC = platform.system() == "Darwin"


def wiggle_mouse():
    x, y = pyautogui.position()
    dx = random.choice([-1, 1]) * WIGGLE_PIXELS
    dy = random.choice([-1, 1]) * WIGGLE_PIXELS
    pyautogui.moveRel(dx, dy, duration=0.4)
    pyautogui.moveTo(x, y, duration=0.4)


def next_tab():
    # Ctrl+Tab works in Chrome/Edge/Firefox on Windows, Linux and macOS
    pyautogui.hotkey("ctrl", "tab")


def main():
    print(f"Mode '{MODE}' every {INTERVAL_SECONDS // 60} min. Ctrl+C to stop.")
    count = 0
    try:
        while True:
            time.sleep(INTERVAL_SECONDS)
            if MODE in ("mouse", "both"):
                wiggle_mouse()
            if MODE in ("tab", "both"):
                next_tab()
            count += 1
            print(f"[{datetime.now():%H:%M:%S}] action #{count} done")
    except KeyboardInterrupt:
        print(f"\nStopped after {count} actions.")


if __name__ == "__main__":
    main()