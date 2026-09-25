# -*- coding: utf-8 -*-
"""
Демо/тест DRCE-LOC на Python с интерактивной настройкой параметров (GUI).

Аналог main.cpp. Режимы входа:
  1. Аргумент — путь к 14/16-битному PNG:   python main.py input.png
  2. Без аргументов — синтетическая HDR-сцена (статическая).
  3. Флаг --camera — захват с веб-камеры в реальном времени:
       python main.py --camera
       Камера: BGR 8-bit -> RGB -> фотопическая яркость -> 14-bit (uint16) -> DRCE-LOC.

Клавиши управления (статический режим, терминал):
   a/d   – bias           (±1)
   w/x   – lambda         (±1)
   k/j   – k1             (±0.1)
   i/u   – k2             (±0.1)
   o/p   – dde            (±0.1)
   l/m   – bright         (±1)
   f/h   – delta2         (±10)
   t/g   – blockRows/Cols (±1 блок, размер ~ N/M)
   r     – сброс к дефолту
   s     – сохранить out_drce_loc.png с текущими параметрами
   Esc   – выход

Клавиши управления (режим камеры):
   Esc   – выход
   s     – сохранить обработанный кадр как out_camera_frame.png
   r     – сброс параметров к дефолту
"""

import sys

import cv2
import numpy as np

from drce_loc import DRCELOC


def fmt(v, width):
    s = f"{v:.3f}"
    if len(s) > width:
        s = s[:width]
    else:
        s = s.ljust(width)
    return s


# Коэффициенты фотопической (дневной) светочувствительности по каналам (ITU-R BT.601).
CR, CG, CB = 0.299, 0.587, 0.114


def bgr_to_gray14(frame_bgr):
    """BGR(8-bit) -> фотопическая яркость -> 14-bit (uint16).

    Brightness = (R*CR*16384 + G*CG*16384 + B*CB*16384) / 256
               == luminance * 64.
    Яркость считается сразу в 14-битном диапазоне без промежуточной
    8-битной квантизации (как в cvtColor RGB2GRAY), что даёт плавный переход.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    lum = (rgb[..., 0] * CR * 16384.0
           + rgb[..., 1] * CG * 16384.0
           + rgb[..., 2] * CB * 16384.0) / 256.0
    return np.clip(np.round(lum), 0, 16383).astype(np.uint16)


def make_synthetic_hdr_scene(rows=768, cols=1024):
    # Эмуляция 14-битного сенсора (диапазон исходных данных 0..16383).
    ys = np.arange(rows)                       # строки
    xs = np.arange(cols)                       # столбцы
    yy, xx = np.meshgrid(ys, xs, indexing="ij")        # (rows, cols): yy=строка, xx=столбец

    sky_val = 16000.0 - 10.0 * yy - 5.0 * xx          # спадает к горизонту и вправо
    scene = np.maximum(5000.0, sky_val)

    horizon = int(rows * 45 / 100)

    def fill(i0, i1, j0, j1, win_hi, win_lo, row_step, col_step):
        r = (np.arange(i0, i1)[:, None] - horizon)    # (r, 1) — строки
        c = np.arange(j0, j1)[None, :]                # (1, c) — столбцы
        base = 500.0 + 100.0 * np.sin(c) * np.sin(c)  # зависит только столбца
        window_row = (r % row_step) < 10
        window_col = (c % col_step) < 14
        win = np.where(window_row & window_col, win_hi, win_lo)   # (r, c)
        scene[i0:i1, j0:j1] = base + win

    fill(horizon, rows, int(cols * 0.3), int(cols * 0.6), 1300.0, 200.0, 18, 22)
    fill(int(horizon * 1.1), rows, int(cols * 0.2), int(cols * 0.4), 1700.0, 300.0, 15, 18)
    fill(int(horizon * 1.2), rows, int(cols * 0.5), int(cols * 0.8), 1600.0, 250.0, 15, 20)

    def circle_grid(img, center, radius, value):
        cy, cx = center
        r = (np.arange(img.shape[0])[:, None] - cy) ** 2   # строки
        c = (np.arange(img.shape[1])[None, :] - cx) ** 2   # столбцы
        mask = (r + c) <= radius ** 2                       # (rows, cols)
        img[mask] = value

    circle_grid(scene, (int(rows * 0.15), int(cols * 0.75)), 2, 16000.0)
    circle_grid(scene, (int(rows * 0.12), int(cols * 0.80)), 1, 15500.0)
    circle_grid(scene, (int(rows * 0.03), int(cols * 0.10)), 1, 14500.0)

    scene += np.random.randn(*scene.shape).astype(np.float64) * 40.0

    scene = np.clip(scene, 0.0, 16383.0)
    return scene.astype(np.uint16)


def save_norm(path, m):
    vis = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
    vis = np.round(vis).astype(np.uint8)
    cv2.imwrite(path, vis)


def default_params(src):
    p = DRCELOC.Params()
    p.blockRows = max(1, src.shape[0] // 16)
    p.blockCols = max(1, src.shape[1] // 16)
    p.delta2 = 3600.0
    p.bias = 100.0
    p.lambda_ = 200.0
    p.k1 = 1.0
    p.k2 = 0.4
    p.dde = 1.0
    p.bright = 128.0
    return p


def main():
    camera_mode = len(sys.argv) > 1 and sys.argv[1] == "--camera"

    src = None
    cap = None

    if camera_mode:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру (index 0).", file=sys.stderr)
            return 1
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        print("Режим камеры: BGR 8-bit -> RGB -> фотопическая яркость -> 14-bit (uint16)")

        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            print("Не удалось прочитать начальный кадр с камеры.", file=sys.stderr)
            cap.release()
            return 1
        src = bgr_to_gray14(frame_bgr)
    elif len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

    if not camera_mode and (src is None or src.size == 0):
        print("Входное изображение не задано/не открыто — генерирую синтетическую 14-битную сцену.")
        src = make_synthetic_hdr_scene()
        cv2.imwrite("input_synthetic_14bit.png", src)

    print(f"Входное изображение: {src.shape[1]}x{src.shape[0]}, "
          f"dtype={src.dtype}")

    # --- Baseline: наивное линейное сжатие ДД (AGC-подобное) ---
    b8 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("out_baseline_linear.png", b8)

    # --- Baseline: OpenCV CLAHE для сравнения (Section 3.1 статьи использует его) ---
    s8 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_out = clahe.apply(s8)
    cv2.imwrite("out_baseline_clahe.png", clahe_out)

    # --- DRCE-LOC с интерактивной настройкой ---
    params = default_params(src)
    block_div = 64

    def recompute():
        t0 = __import__("time").perf_counter()
        out = DRCELOC(params).process(src)
        t1 = __import__("time").perf_counter()
        ms = (t1 - t0) * 1000.0
        display = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return out, display, ms

    _, display, ms = recompute()

    window_name = "DRCE-LOC Interactive"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        if camera_mode:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            src = np.clip(gray.astype(np.int32) * 64, 0, 16383).astype(np.uint16)

        lines = [
            "=== DRCE-LOC ===",
            "bias   [a][d]= " + fmt(params.bias, 6),
            "lambda [w][x]= " + fmt(params.lambda_, 6),
            "k1     [k][j]= " + fmt(params.k1, 4),
            "k2     [i][u]= " + fmt(params.k2, 4),
            "dde    [o][p]= " + fmt(params.dde, 4),
            "bright [l][m]= " + fmt(params.bright, 6),
            "delta2 [f][h]= " + fmt(params.delta2, 6),
            "block  [t][g]= " + str(src.shape[0] // block_div),
            "----------------",
            "time   = " + f"{int(ms):04d} ms",
        ]

        vis = display.copy()
        y = 20
        for line in lines:
            cv2.putText(vis, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y += 16

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1 if camera_mode else 0) & 0xFF
        changed = False

        if key == 27:  # Esc — выход
            if camera_mode:
                cap.release()
            cv2.imwrite("out_drce_loc.png", display)
            break

        elif key == 115:  # 's' — сохранить результат
            if camera_mode:
                cv2.imwrite("out_camera_frame.png", display)
                print("Сохранено: out_camera_frame.png")
            else:
                cv2.imwrite("out_drce_loc.png", display)
                print("Сохранено: out_drce_loc.png")

        elif key == 114:  # 'r' — сброс к дефолту
            block_div = 64
            params.blockRows = max(1, src.shape[0] // block_div)
            params.blockCols = max(1, src.shape[1] // block_div)
            params.bias = 100.0
            params.lambda_ = 200.0
            params.k1 = 1.0
            params.k2 = 0.4
            params.dde = 1.0
            params.bright = 128.0
            params.delta2 = -1.0
            changed = True

        elif key == ord('a'):
            params.bias -= 1.0; changed = True
        elif key == ord('d'):
            params.bias += 1.0; changed = True
        elif key == ord('w'):
            params.lambda_ += 1.0; changed = True
        elif key == ord('x'):
            params.lambda_ -= 1.0; changed = True
        elif key == ord('k'):
            params.k1 += 0.1; changed = True
        elif key == ord('j'):
            params.k1 -= 0.1; changed = True
        elif key == ord('i'):
            params.k2 += 0.1; changed = True
        elif key == ord('u'):
            params.k2 -= 0.1; changed = True
        elif key == ord('o'):
            params.dde += 0.1; changed = True
        elif key == ord('p'):
            params.dde -= 0.1; changed = True
        elif key == ord('l'):
            params.bright += 1.0; changed = True
        elif key == ord('m'):
            params.bright -= 1.0; changed = True
        elif key == ord('f'):
            params.delta2 -= 10.0; changed = True
        elif key == ord('h'):
            params.delta2 += 10.0; changed = True
        elif key == ord('t'):
            block_div = max(2, block_div + 1)
            params.blockRows = max(1, src.shape[0] // block_div)
            params.blockCols = max(1, src.shape[1] // block_div)
            changed = True
        elif key == ord('g'):
            block_div = max(2, block_div - 1)
            params.blockRows = max(1, src.shape[0] // block_div)
            params.blockCols = max(1, src.shape[1] // block_div)
            changed = True

        if changed or camera_mode:
            _, display, ms = recompute()
            if changed:
                print(f"params: bias={params.bias} lambda={params.lambda_} "
                      f"k1={params.k1} k2={params.k2} dde={params.dde} "
                      f"bright={params.bright} t={ms:.1f}ms")

    if camera_mode:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless-версия OpenCV без GUI-бэкенда
    return 0


if __name__ == "__main__":
    sys.exit(main())
