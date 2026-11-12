"""寿司打 自動入力ツール

画面上のローマ字表示（例: "saikousinkiroku"）を OCR で読み取り、
キーボード入力として送信します。

使い方:
    python sushida_auto.py --calibrate   # ローマ字が表示される範囲をドラッグで指定
    python sushida_auto.py --test        # 1回だけ読み取って結果を表示（入力はしない）
    python sushida_auto.py               # 自動入力を開始

ホットキー:
    F9  : 開始 / 一時停止
    F10 : 終了
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import mss
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pynput import keyboard

CONFIG_PATH = Path(__file__).with_name("config.json")

DEFAULT_CONFIG = {
    # ローマ字の行の範囲（画面の物理ピクセル座標）
    "region": None,  # {"left": int, "top": int, "width": int, "height": int}
    # Tesseract 実行ファイルのパス（PATH に通っていれば null のままでOK）
    "tesseract_cmd": None,
    # 白文字とみなす明るさのしきい値 (0-255)。
    # 入力済みの文字は灰色になるため、これより暗い文字は無視される。
    "threshold": 200,
    # OCR 前の拡大倍率
    "scale": 3,
    # 1文字ごとの入力間隔（秒）
    "key_interval": 0.01,
    # 画面を読み取る間隔（秒）
    "poll_interval": 0.05,
    # 同じ文字列を連続で読んだとき、再入力するまでの待ち時間（秒）
    "retype_after": 0.6,
}

# 寿司打のローマ字に出てくる文字
WHITELIST = "abcdefghijklmnopqrstuvwxyz0123456789-,.!?"
VALID_RE = re.compile(r"[^a-z0-9\-,.!?]")


# ---------------------------------------------------------------- 設定

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    if cfg.get("tesseract_cmd"):
        pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]
    elif sys.platform == "win32":
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def enable_dpi_awareness() -> None:
    """Windows の拡大表示設定でも、座標を物理ピクセルにそろえる。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ---------------------------------------------------------------- 範囲指定

def calibrate(cfg: dict) -> None:
    """全画面の半透明ウィンドウを出し、ドラッグで範囲を選ばせる。"""
    import tkinter as tk

    print("ローマ字の行（白い英字の部分）をドラッグで囲んでください。Esc でキャンセル。")
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    root.attributes("-topmost", True)
    root.configure(cursor="crosshair")
    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    state: dict = {"start": None, "rect": None, "result": None}

    def on_press(e):
        state["start"] = (e.x_root, e.y_root)
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=3)

    def on_drag(e):
        if not state["start"]:
            return
        sx, sy = state["start"]
        ox, oy = root.winfo_rootx(), root.winfo_rooty()
        canvas.coords(state["rect"], sx - ox, sy - oy, e.x, e.y)

    def on_release(e):
        sx, sy = state["start"]
        x1, x2 = sorted((sx, e.x_root))
        y1, y2 = sorted((sy, e.y_root))
        if x2 - x1 > 10 and y2 - y1 > 5:
            state["result"] = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
            root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()

    if state["result"]:
        cfg["region"] = state["result"]
        save_config(cfg)
        print(f"範囲を保存しました: {state['result']} -> {CONFIG_PATH}")
    else:
        print("キャンセルしました。")


# ---------------------------------------------------------------- OCR

def preprocess(img: Image.Image, cfg: dict) -> Image.Image:
    """白い文字だけを取り出し、白地に黒文字の画像にする。"""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    # 白 = RGB すべてが明るい。黄色や灰色（入力済み文字）は除外される。
    mask = (arr.min(axis=2) >= cfg["threshold"])
    out = np.where(mask, 0, 255).astype(np.uint8)
    pil = Image.fromarray(out, mode="L")
    s = cfg["scale"]
    if s and s != 1:
        pil = pil.resize((pil.width * s, pil.height * s), Image.LANCZOS)
    return ImageOps.expand(pil, border=20, fill=255)


def read_romaji(sct, cfg: dict, debug: bool = False) -> str:
    shot = sct.grab(cfg["region"])
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    proc = preprocess(img, cfg)
    if debug:
        img.save("debug_raw.png")
        proc.save("debug_processed.png")
    text = pytesseract.image_to_string(
        proc,
        config=f"--psm 7 --oem 3 -c tessedit_char_whitelist={WHITELIST}",
    )
    return VALID_RE.sub("", text.strip().lower())


# ---------------------------------------------------------------- 本体

class AutoTyper:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.running = False
        self.quit = threading.Event()
        self.kb = keyboard.Controller()

    def type_text(self, text: str) -> None:
        for ch in text:
            if not self.running or self.quit.is_set():
                return
            self.kb.press(ch)
            self.kb.release(ch)
            time.sleep(self.cfg["key_interval"])

    def on_key(self, key):
        if key == keyboard.Key.f9:
            self.running = not self.running
            print("▶ 開始" if self.running else "⏸ 一時停止")
        elif key == keyboard.Key.f10:
            print("終了します。")
            self.quit.set()
            return False

    def run(self) -> None:
        listener = keyboard.Listener(on_press=self.on_key)
        listener.start()
        print("ブラウザの寿司打をクリックしてフォーカスを合わせ、F9 で開始 / F10 で終了。")

        last_text, last_time = "", 0.0
        with mss.mss() as sct:
            while not self.quit.is_set():
                if not self.running:
                    time.sleep(0.05)
                    continue
                text = read_romaji(sct, self.cfg)
                now = time.time()
                if len(text) < 1:
                    time.sleep(self.cfg["poll_interval"])
                    continue
                # 同じ文字列をすぐ再入力しない（画面の更新待ち）
                if text == last_text and now - last_time < self.cfg["retype_after"]:
                    time.sleep(self.cfg["poll_interval"])
                    continue
                print(f"入力: {text}")
                self.type_text(text)
                last_text, last_time = text, time.time()
                time.sleep(self.cfg["poll_interval"])
        listener.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="寿司打 自動入力ツール")
    parser.add_argument("--calibrate", action="store_true", help="読み取り範囲を指定する")
    parser.add_argument("--test", action="store_true", help="1回だけ読み取って表示（画像も保存）")
    args = parser.parse_args()

    enable_dpi_awareness()
    cfg = load_config()

    if args.calibrate or not cfg.get("region"):
        calibrate(cfg)
        if not cfg.get("region"):
            return
        if args.calibrate:
            return

    if args.test:
        with mss.mss() as sct:
            text = read_romaji(sct, cfg, debug=True)
        print(f"読み取り結果: '{text}'")
        print("debug_raw.png / debug_processed.png を保存しました。")
        return

    AutoTyper(cfg).run()


if __name__ == "__main__":
    main()
