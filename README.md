# DRCE-LOC — апробация алгоритма на OpenCV/C++ (+ Python)

Реализация из статьи: Zhu Y. et al., *"A Low-Delay Dynamic Range Compression and Contrast
Enhancement Algorithm Based on an Uncooled Infrared Sensor with Local Optimal Contrast"*,
Sensors 2023, 23, 8860.

## Состав
- `drce_loc.{hpp,cpp}` — класс `DRCELOC`, шаги 1–8 алгоритма (Fig. 2 статьи).
- `main.cpp` — демо: загрузка изображения / синтетическая сцена / **веб-камера**, прогон
  алгоритма, сравнение с линейным AGC и CLAHE, метрики RMS/Entropy/Tenengrad, сохранение
  слоёв `dbg_*.png`.
- `python/` — порт на Python (`drce_loc.py` + `main.py`) с теми же режимами.

## Сборка (C++)
```bash
# напрямую
g++ -O2 -std=c++17 main.cpp drce_loc.cpp $(pkg-config --cflags --libs opencv4) -o drce_loc_demo
# или CMake
mkdir build && cd build && cmake .. && make -j
```

## Запуск
```bash
./drce_loc_demo путь/к/ик_кадру.png     # 8/16-бит, одноканальное
./drce_loc_demo                          # синтетическая тестовая сцена
./drce_loc_demo --camera                 # веб-камера: BGR 8-bit -> фотопическая яркость -> 14-bit (CV_16U)
LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 ./drce_loc_demo   # для многопоточности

# Python (нужны numpy + opencv-python)
cd python && python3 main.py input.png        # файл
python3 main.py                               # синтетическая сцена
python3 main.py --camera                      # веб-камера
```

Клавиши в окне: `a/d` bias, `w/x` lambda, `k/j` k1, `i/u` k2, `o/p` dde, `l/m` bright,
`f/h` delta2, `t/g` блок, `r` сброс, `s` сохранить, `Esc` выход.

## Ключевые параметры (DRCELOC::Params)
| Поле              | Формула статьи | Рекомендация      |
|-------------------|----------------|---------------------|
| blockRows/Cols    | X, Y (Step 1)  | ≈ 64×64 px          |
| bias              | Bias, Eq. 1    | 128–384             |
| delta2            | δ², Eq. 2–3    | ≥ (M/X)·(N/Y)       |
| guidedRadius/Eps  | w, ε, Eq. 4    | окно 5×5 (radius=2) |
| lambda            | λ, Eq. 5       | 128–384             |
| k1                | k1, Eq. 6      | 1.0                 |
| k2                | k2, Eq. 6      | 0.2–0.7             |
| gLow/gHigh, dde   | Eq. 7–8        | см. `drce_loc.hpp`  |

**Важно:** `normalizeInputTo255=true` по умолчанию — вход (даже 14/16-бит) линейно
нормализуется в 0..255, так как диапазоны Bias/λ из статьи заданы для этой шкалы.

## Адаптация к сцене
| Параметр | Насыщенная сцена | Малые цели | Слабый контраст |
|----------|------------------|------------|-----------------|
| Bias     | 128–192          | 320–384    | 200–280         |
| k1       | 1.0              | 1.0–1.2    | 1.0             |
| k2       | 0.5–0.7          | 0.2–0.4    | 0.4–0.6         |
| Блок     | 64×64            | 32×32      | 64×64 / 128×128 |
| δ²       | ≥ 4096           | ≥ 1024     | ≥ 4096          |
| λ        | 300–384          | 200–300    | 150–250         |
| DDE      | 1.0–1.5          | 2.0–3.0    | 1.5–2.0         |
| g_L      | 0.6–0.8          | 0.3–0.5    | 0.5–0.7         |

## Известные упрощения относительно статьи
1. **Eq. 7** (шумовая маска детального слоя) — в статье не приведена полностью; реализована
   типовая интерпретация «меньше гейн в однородных зонах».
2. **FPGA-оптимизации** (Section 3.2: 16/25 соседей, LUT экспоненты, фикс. точка) — не
   воспроизведены, так как влияют только на hardware-производительность.
3. Разделимый Gaussian upsampling (Eq. 2–3) сделан через `cv::gemm` — точный, но быстрее наивного.

## Режим камеры
Веб-камера даёт BGR 8-bit; конвертация в 14-bit выполняется взвешенной фотопической
яркостью: `Brightness = (R·c_r + G·c_g + B·c_b)·16384/256`, c_r=0.299, c_g=0.587, c_b=0.114
(ITU-R BT.601). Расчёт сразу в 14-bit, без промежуточной 8-битной квантизации.
