// Демо/тест DRCE-LOC с интерактивной настройкой параметров (GUI).
//
// Режимы входа:
//   1. Аргумент — путь к 14/16-битному PNG:  ./drce_loc_demo input.png
//   2. Без аргументов — синтетическая HDR-сцена (статическая).
//   3. Флаг --camera — захват с веб-камеры в реальном времени:
//      ./drce_loc_demo --camera
//      Камера: BGR 8-bit -> grayscale -> 14-bit (CV_16U) -> DRCE-LOC.
//
// Клавиши управления (статический режим, терминал):
//   a/d   – bias           (±1)
//   w/x   – lambda         (±1)
//   k/j   – k1             (±0.1)
//   i/u   – k2             (±0.1)
//   o/p   – dde            (±0.1)
//   l/m   – bright         (±1)
//   f/h   – delta2         (±10)
//   t/g   – blockRows/Cols (±1 блок, размер ~ N/M)
//   r     – сброс к дефолту
//   s     – сохранить out_drce_loc.png с текущими параметрами
//   Esc   – выход
//
// Клавиши управления (режим камеры):
//   Esc   – выход
//   s     – сохранить обработанный кадр как out_camera_frame.png
//   r     – сброс параметров к дефолту


#include "drce_loc.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Коэффициенты фотопической (дневной) светочувствительности по каналам (ITU-R BT.601).
static const double kCr = 0.299;
static const double kCg = 0.587;
static const double kCb = 0.114;

static std::string fmt(double v, int width) {
    std::ostringstream ss;
    ss.precision(3);
    ss << std::fixed << v;
    std::string s = ss.str();
    if (s.size() > (size_t)width) s.resize(width);
    else s.resize(width, ' ');
    return s;
}

static cv::Mat makeSyntheticHDRScene(int rows = 768, int cols = 1024) {
    // Эмуляция 14-битного сенсора (диапазон исходных данных 0..16383),
    // хранится в CV_16U.
    cv::Mat scene(rows, cols, CV_32F);

    // Яркое "небо" сверху с плавным градиентом (высокий сигнал).
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float skyVal = 16000.0f - 10.0f * i - 5.0f * j; // спадает к горизонту и вправо
            scene.at<float>(i, j) = std::max(5000.0f, skyVal);
        }
    }

    // Три тёмных "здания" снизу с оконной текстурой (низкий сигнал, высокая деталь).
    int horizon = rows * 45 / 100;

    for (int i = horizon; i < rows; ++i) {
        for (int j = cols*0.3; j < cols*0.6; ++j) {
            float base = 500.0f + 100.0f * std::sin(j * 0.45) * std::sin(j * 0.45);
            bool windowRow = ((i - horizon) % 18) < 10;
            bool windowCol = (j % 22) < 14;
            float win = (windowRow && windowCol) ? 1300.0f : 200.0f; // реалистичная амплитуда детали на низком сигнале
            scene.at<float>(i, j) = base + win;
        }
    }
    for (int i = horizon*1.1; i < rows; ++i) {
        for (int j = cols*0.2; j < cols*0.4; ++j) {
            float base = 600.0f + 150.0f * std::sin(j * 0.35) * std::sin(j * 0.35);
            bool windowRow = ((i - horizon) % 15) < 10;
            bool windowCol = (j % 18) < 14;
            float win = (windowRow && windowCol) ? 1700.0f : 300.0f; // реалистичная амплитуда детали на низком сигнале
            scene.at<float>(i, j) = base + win;
        }
    }

    for (int i = horizon*1.2; i < rows; ++i) {
        for (int j = cols*0.5; j < cols*0.8; ++j) {
            float base = 700.0f + 170.0f * std::sin(j * 0.35) * std::sin(j * 0.35);
            bool windowRow = ((i - horizon) % 15) < 10;
            bool windowCol = (j % 20) < 14;
            float win = (windowRow && windowCol) ? 1600.0f : 250.0f; // реалистичная амплитуда детали на низком сигнале
            scene.at<float>(i, j) = base + win;
        }
    }

    // Мелкие яркие цели на фоне неба (аналог "small target scene").
    cv::circle(scene, cv::Point(cols * 0.75, rows * 0.15), 2, cv::Scalar(16000), -1);
    cv::circle(scene, cv::Point(cols * 0.80, rows * 0.12), 1, cv::Scalar(15500), -1);
    cv::circle(scene, cv::Point(cols * 0.10, rows * 0.03), 1, cv::Scalar(14500), -1);

    // Шум сенсора.
    cv::Mat noise(rows, cols, CV_32F);
    cv::randn(noise, 0, 40.0);
    scene += noise;

    cv::Mat scene16u;
    cv::max(scene, 0, scene);
    cv::min(scene, 16383, scene);
    scene.convertTo(scene16u, CV_16U);
    return scene16u;
}

static void saveNorm(const std::string& path, const cv::Mat& m32f_or_64f) {
    cv::Mat vis;
    cv::normalize(m32f_or_64f, vis, 0, 255, cv::NORM_MINMAX);
    vis.convertTo(vis, CV_8U);
    cv::imwrite(path, vis);
}

int main(int argc, char** argv) {
    bool cameraMode = (argc > 1 && std::string(argv[1]) == "--camera");
    cv::Mat src;
    cv::VideoCapture cap;
    cv::Mat frameBGR, rgbFrame, gray8u;

    if (cameraMode) {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Не удалось открыть камеру (index 0).\n";
            return 1;
        }
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768);
        std::cout << "Режим камеры: BGR 8-bit -> RGB -> grayscale -> 14-bit (CV_16U)\n";

        // Читаем начальный кадр, чтобы заполнить src до секции baseline.
        cap >> frameBGR;
        if (frameBGR.empty()) {
            std::cerr << "Не удалось прочитать начальный кадр с камеры.\n";
            return 1;
        }
        cv::cvtColor(frameBGR, rgbFrame, cv::COLOR_BGR2RGB);
        // Brightness = (R*Cr + G*Cg + B*Cb) * 16384 / 256 == luminance * 64.
        // Фотопическая яркость сразу в 14-bit, без 8-битной квантизации.
        cv::Mat rgbF;
        rgbFrame.convertTo(rgbF, CV_64F);   // CV_64FC3 (R,G,B)
        std::vector<cv::Mat> channels;
        cv::split(rgbF, channels);          // каждый канал — отдельный CV_64UC1
        cv::Mat lum = (channels[0] * kCr + channels[1] * kCg + channels[2] * kCb) * 64.0;
        lum.convertTo(src, CV_16U);
    }

    if (!cameraMode && argc > 1) {
        src = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    }
    if (!cameraMode && src.empty()) {
        std::cout << "Входное изображение не задано/не открыто — "
                     "генерирую синтетическую 14-битную сцену.\n";
        src = makeSyntheticHDRScene();
        cv::imwrite("input_synthetic_14bit.png", src);
    }
    std::cout << "Входное изображение: " << src.cols << "x" << src.rows
              << ", depth=" << src.depth() << ", channels=" << src.channels() << "\n";

    // --- Baseline для сравнения: наивное линейное сжатие ДД (AGC-подобное) ---
    cv::Mat baseline8u;
    cv::normalize(src, baseline8u, 0, 255, cv::NORM_MINMAX);
    baseline8u.convertTo(baseline8u, CV_8U);
    cv::imwrite("out_baseline_linear.png", baseline8u);

    // --- Baseline: OpenCV CLAHE для сравнения (Section 3.1 статьи использует его) ---
    cv::Mat src8u_for_clahe;
    cv::normalize(src, src8u_for_clahe, 0, 255, cv::NORM_MINMAX);
    src8u_for_clahe.convertTo(src8u_for_clahe, CV_8U);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat claheOut;
    clahe->apply(src8u_for_clahe, claheOut);
    cv::imwrite("out_baseline_clahe.png", claheOut);

    // --- DRCE-LOC с интерактивной настройкой ---
    DRCELOC::Params params;
    // Размер блока ~64x64 при 1024x768, как рекомендует Section 2.4 статьи.
    params.blockRows = std::max(1, src.rows / 16);
    params.blockCols = std::max(1, src.cols / 16);
    params.delta2   = 3600.0;
    params.bias     = 100.0;
    params.lambda   = 200.0;
    params.k1       = 1.0;
    params.k2       = 0.4;
    params.dde      = 1.0;
    params.bright   = 128.0;

    int blockDiv = 64;  // делитель размера блока (меняется [t]/[g])

    auto t0 = std::chrono::high_resolution_clock::now();
    DRCELOC algo(params);
    cv::Mat out = algo.process(src);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat display;
    cv::normalize(out, display, 0, 255, cv::NORM_MINMAX);
    display.convertTo(display, CV_8U);

    const char* windowName = "DRCE-LOC Interactive";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    while (true) {
        if (cameraMode) {
            cap >> frameBGR;
            if (frameBGR.empty()) break;
            cv::cvtColor(frameBGR, rgbFrame, cv::COLOR_BGR2RGB);
            // Brightness = (R*Cr + G*Cg + B*Cb) * 16384 / 256 == luminance * 64.
            // Фотопическая яркость сразу в 14-bit, без 8-битной квантизации.
            cv::Mat rgbF;
            rgbFrame.convertTo(rgbF, CV_64F);   // CV_64FC3 (R,G,B)
            std::vector<cv::Mat> channels;
            cv::split(rgbF, channels);          // каждый канал — отдельный CV_64UC1
            cv::Mat lum = (channels[0] * kCr + channels[1] * kCg + channels[2] * kCb) * 64.0;
            lum.convertTo(src, CV_16U);
        }

        // Формируем вертикальный список параметров слева
        std::vector<std::string> lines = {
            "=== DRCE-LOC ===",
            "bias   [a][d]= " + fmt(params.bias, 6),
            "lambda [w][x]= " + fmt(params.lambda, 6),
            "k1     [k][j]= " + fmt(params.k1, 4),
            "k2     [i][u]= " + fmt(params.k2, 4),
            "dde    [o][p]= " + fmt(params.dde, 4),
            "bright [l][m]= " + fmt(params.bright, 6),
            "delta2 [f][h]= " + fmt(params.delta2, 6),
            "block  [t][g]= " + std::to_string(src.rows / blockDiv),
            "----------------",
            "time   = " + std::to_string((int)ms).substr(0, 4) + " ms"
        };

        cv::Mat vis = display.clone();
        int y = 20;
        for (const auto& line : lines) {
            cv::putText(vis, line, cv::Point(10, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
            y += 16;
        }

        cv::imshow(windowName, vis);

        int key = cv::waitKey(cameraMode ? 1 : 0) & 0xFF;
        bool changed = false;

        switch (key) {
            case 27:  // Esc — выход
                cv::destroyAllWindows();
                if (cameraMode) cap.release();
                cv::imwrite("out_drce_loc.png", display);
                return 0;

            case 115: // 's' — сохранить результат
                if (cameraMode) {
                    cv::imwrite("out_camera_frame.png", display);
                    std::cout << "Сохранено: out_camera_frame.png\n";
                } else {
                    cv::imwrite("out_drce_loc.png", display);
                    std::cout << "Сохранено: out_drce_loc.png\n";
                }
                break;

            case 114: // 'r' — сброс к дефолту
                blockDiv          = 64;
                params.blockRows  = std::max(1, src.rows / blockDiv);
                params.blockCols  = std::max(1, src.cols / blockDiv);
                params.bias       = 100.0;
                params.lambda     = 200.0;
                params.k1         = 1.0;
                params.k2         = 0.4;
                params.dde        = 1.0;
                params.bright     = 128.0;
                params.delta2     = -1.0;
                changed = true;
                break;

            case 'a':   // bias -1
                params.bias     -= 1.0;  changed = true; break;
            case 'd':   // bias +1
                params.bias     += 1.0;  changed = true; break;
            case 'w':   // lambda +1
                params.lambda   += 1.0;  changed = true; break;
            case 'x':   // lambda -1
                params.lambda   -= 1.0;  changed = true; break;
            case 'k':   // k1 +0.1
                params.k1       += 0.1f; changed = true; break;
            case 'j':   // k1 -0.1
                params.k1       -= 0.1f; changed = true; break;
            case 'i':   // k2 +0.1
                params.k2       += 0.1f; changed = true; break;
            case 'u':   // k2 -0.1
                params.k2       -= 0.1f; changed = true; break;
            case 'o':   // dde +0.1
                params.dde      += 0.1f; changed = true; break;
            case 'p':   // dde -0.1
                params.dde      -= 0.1f; changed = true; break;
            case 'l':   // bright +1  (key right of 'l' on US keyboard)
                params.bright   += 1.0f; changed = true; break;
            case 'm':  // bright -1  (apostrophe, next to ';')
                params.bright   -= 1.0f; changed = true; break;
            case 'f':   // delta2 -1.0
                params.delta2   -= 10.0f;  changed = true; break;
            case 'h':   // delta2 +1.0
                params.delta2   += 10.0f;  changed = true; break;
            case 't':   // blockRows/Cols -1 (увеличить блок)
                blockDiv        = std::max(2, blockDiv + 1);
                params.blockRows = std::max(1, src.rows / blockDiv);
                params.blockCols = std::max(1, src.cols / blockDiv);
                changed = true; break;
            case 'g':   // blockRows/Cols +1 (уменьшить блок)
                blockDiv        = std::max(2, blockDiv - 1);
                params.blockRows = std::max(1, src.rows / blockDiv);
                params.blockCols = std::max(1, src.cols / blockDiv);
                changed = true; break;

            default:
                // Непознанная клавиша — игнорируем
                break;
        }

        if (changed || cameraMode) {
            t0 = std::chrono::high_resolution_clock::now();
            DRCELOC algo2(params);
            out = algo2.process(src);
            t1 = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            cv::normalize(out, display, 0, 255, cv::NORM_MINMAX);
            display.convertTo(display, CV_8U);

            if (changed) {
                std::cout << "params: bias=" << params.bias
                          << " lambda=" << params.lambda
                          << " k1=" << params.k1
                          << " k2=" << params.k2
                          << " dde=" << params.dde
                          << " bright=" << params.bright
                          << " t=" << ms << "ms\n";
            }
        }
    }
}