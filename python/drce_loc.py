# -*- coding: utf-8 -*-
"""
Порт DRCE-LOC на Python (numpy + OpenCV).

Реализация по статье:
  Zhu Y., Zhou Y., Jin W., Zhang L., Wu G., Shao Y.
  "A Low-Delay Dynamic Range Compression and Contrast Enhancement Algorithm
   Based on an Uncooled Infrared Sensor with Local Optimal Contrast"
  Sensors 2023, 23, 8860.

Соответствие шагам статьи (Section 2.1, Figure 2):
  Step 1  computeBlockStats()      -> Block_mean, Block_std
  Step 2  computeStretchCoeff()    -> Stretch_para  (Eq. 1)
  Step 3  gaussianUpsample()       -> NStretch, NMean (Eq. 2,3)
  Step 4  selfGuidedFilter()       -> Base(GF), Detail, a_coeff (Eq. 4)
  Step 5  computeBrightnessGuide() -> Igc           (Eq. 5)
  Step 6  process(): Base_out                              (Eq. 6)
  Step 7  process(): Detail_out (noise mask)              (Eq. 7)
  Step 8  process(): Iout = Base_out + DDE*Detail_out     (Eq. 8)

Вход: одноканальный массив numpy любой глубины (uint8/uint16/float32).
Выход: CV_8U массив того же размера (numpy uint8).
"""

import cv2
import numpy as np


class DRCELOC:
    class Params:
        def __init__(self):
            # Предварительная нормализация входа в номинальный диапазон 0..255
            # (min-max по кадру) до блочной статистики.
            self.normalizeInputTo255 = True

            # --- Шаг 1-3: блочная статистика и её апсемплинг ---
            self.blockRows = 12     # X: число блоков по вертикали
            self.blockCols = 16     # Y: число блоков по горизонтали
            self.bias = 200.0       # Bias, Eq. 1
            self.delta2 = -1.0      # delta^2, Eq. 2-3 (<=0 -> автоматически)

            # --- Шаг 4: guided filter (self-guided, окно 5x5 -> radius=2) ---
            self.guidedRadius = 2
            self.guidedEps = 100.0  # регуляризация в Eq. 4 (epsilon)

            # --- Шаг 5: глобальная яркостная направляющая карта ---
            self.lambda_ = 200.0    # lambda, Eq. 5
            self.bright = 128.0     # Bright, Eq. 5

            # --- Шаг 6: компрессия ДД + контрастное усиление ---
            self.k1 = 1.0           # вес локального контраста, Eq. 6
            self.k2 = 0.4           # вес глобального контраста / подавление гало, Eq. 6

            # --- Шаг 7: шумовая маска детального слоя ---
            self.gLow = 0.3         # минимальный gain в однородных зонах
            self.gHigh = 1.2        # максимальный gain в детализированных зонах

            # --- Шаг 8: финальный синтез ---
            self.dde = 1.0          # Detail enhancement factor, Eq. 8

    def __init__(self, params=None):
        self.params = params if params is not None else DRCELOC.Params()
        self.dbg = {}

    @staticmethod
    def _box_filter(img32f, ksize):
        # Прямый аналог cv::boxFilter(src, CV_32F, ksize).
        return cv2.boxFilter(img32f, -1, ksize)

    def compute_block_stats(self, img32f):
        M, N = img32f.shape[:2]
        X, Y = self.params.blockRows, self.params.blockCols
        block_mean = np.zeros((X, Y), dtype=np.float64)
        block_std = np.zeros((X, Y), dtype=np.float64)
        for bx in range(X):
            r0 = int(round(bx * M / X))
            r1 = int(round((bx + 1) * M / X))
            r1 = max(r1, r0 + 1)
            for by in range(Y):
                c0 = int(round(by * N / Y))
                c1 = int(round((by + 1) * N / Y))
                c1 = max(c1, c0 + 1)
                roi = img32f[r0:r1, c0:c1]
                block_mean[bx, by] = roi.mean()
                block_std[bx, by] = roi.std()  # population std (ddof=0), как meanStdDev
        return block_mean, block_std

    def gaussian_upsample(self, block_map, M, N, delta2):
        X, Y = block_map.shape[:2]
        row_centers = np.array([(bx + 0.5) * M / X for bx in range(X)])
        col_centers = np.array([(by + 0.5) * N / Y for by in range(Y)])

        i = np.arange(M)[:, None]          # M x 1
        j = np.arange(N)[:, None]          # N x 1
        wrow = np.exp(-((i - row_centers) ** 2) / delta2)   # M x X
        wcol = np.exp(-((j - col_centers) ** 2) / delta2)   # N x Y

        numerator = (wrow @ block_map) @ wcol.T              # M x N
        row_sum = wrow.sum(axis=1)[:, None]                  # M x 1
        col_sum = wcol.sum(axis=1)[None, :]                  # 1 x N
        denom = row_sum * col_sum                            # M x N (outer product)

        result = numerator / denom
        return result.astype(np.float32)

    def self_guided_filter(self, img32f):
        r = self.params.guidedRadius
        ksize = (2 * r + 1, 2 * r + 1)
        eps = self.params.guidedEps

        mean_i = self._box_filter(img32f, ksize)
        mean_ii = self._box_filter(img32f * img32f, ksize)
        var_i = mean_ii - mean_i * mean_i

        a = var_i / (var_i + eps)
        b = mean_i - a * mean_i

        base = self._box_filter(a, ksize) * img32f + self._box_filter(b, ksize)
        detail = img32f - base
        return base, detail, a

    def compute_brightness_guide(self, img32f):
        mean = float(img32f.mean())
        std = float(img32f.std())
        return (255.0 * (img32f - mean)) / (std + self.params.lambda_) + self.params.bright

    def process(self, src):
        assert src.ndim == 2 and src.size > 0, "DRCELOC: ожидается одноканальное непустое изображение"
        img = src.astype(np.float32)

        if self.params.normalizeInputTo255:
            lo, hi = float(img.min()), float(img.max())
            if hi - lo > 1e-6:
                img = (img - lo) * (255.0 / (hi - lo))

        M, N = img.shape[:2]
        X, Y = self.params.blockRows, self.params.blockCols

        block_mean, block_std = self.compute_block_stats(img)
        block_stretch = 255.0 / (block_std + self.params.bias)   # Eq. 1
        self.dbg['blockMean'] = block_mean
        self.dbg['blockStd'] = block_std
        self.dbg['blockStretch'] = block_stretch

        delta2 = self.params.delta2
        if delta2 <= 0.0:
            block_area = (M / X) * (N / Y)
            delta2 = 1.5 * block_area
        self.dbg['NStretch'] = self.gaussian_upsample(block_stretch, M, N, delta2)
        self.dbg['NMean'] = self.gaussian_upsample(block_mean, M, N, delta2)

        gf_base, gf_detail, gf_a = self.self_guided_filter(img)
        self.dbg['gfBase'] = gf_base
        self.dbg['gfDetail'] = gf_detail
        self.dbg['gfA'] = gf_a

        self.dbg['Igc'] = self.compute_brightness_guide(img)

        base_out = self.params.k1 * self.dbg['NStretch'] * (img - self.dbg['NMean']) \
                   + self.params.k2 * self.dbg['Igc']   # Eq. 6
        self.dbg['baseOut'] = base_out

        abs_a = np.abs(gf_a)
        max_a = float(abs_a.max())
        if max_a < 1e-9:
            max_a = 1.0
        norm_a = abs_a / max_a                       # 0..1
        mask = self.params.gLow + (self.params.gHigh - self.params.gLow) * norm_a
        detail_out = gf_detail * abs_a * mask         # Eq. 7
        self.dbg['detailOut'] = detail_out

        out32f = base_out + self.params.dde * detail_out   # Eq. 8

        clipped = np.clip(out32f, 0.0, 255.0)
        return np.round(clipped).astype(np.uint8)
