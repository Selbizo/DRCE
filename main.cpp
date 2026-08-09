// Демо/тест DRCE-LOC с интерактивной настройкой параметров (GUI).
//
// Клавиши управления (терминал):
//   a/d   – bias           (±1)
//   w/s   – lambda         (±1)
//   k/j   – k1             (±0.1)
//   i/u   – k2             (±0.1)
//   o/p   – dde            (±0.1)
//   l/;   – bright         (±1)
//   r     – сброс к дефолту
//   s     – сохранить out_drce_loc.png с текущими параметрами
//   Esc   – выход
//
// Если путь не задан или файл не открылся — генерируется синтетическая
// сцена с высоким динамическим диапазоном (аналог "rich scene" из статьи:
// яркое небо + тёмное здание + мелкие цели), сохранённая как 16-битный
// PNG, чтобы было на чём проверить сжатие ДД 14/16 бит -> 8 бит.


#include "drce_loc.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    cv::Mat src;

    if (argc > 1) {
        src = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    }
    if (src.empty()) {
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
    params.blockRows = std::max(1, src.rows / 32);
    params.blockCols = std::max(1, src.cols / 32);
    params.delta2   = -1.0;
    params.bias     = 100.0;
    params.lambda   = 200.0;
    params.k1       = 1.0;
    params.k2       = 0.4;
    params.dde      = 1.0;
    params.bright   = 128.0;

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
        // Формируем строку с текущими параметрами
        std::string info = "bias [a,d]=" + std::to_string(params.bias).substr(0, 5) +
                           " lambda [w,x]=" + std::to_string(params.lambda).substr(0, 5) +
                           " k1 [k,j]=" + std::to_string(params.k1).substr(0, 4) +
                           " k2 [i,u]=" + std::to_string(params.k2).substr(0, 4) +
                           " dde [o,p]=" + std::to_string(params.dde).substr(0, 4) +
                           " bright [;,']=" + std::to_string(params.bright).substr(0, 5) +
                           " | t=" + std::to_string(ms).substr(0, 5) + "ms";

        cv::Mat vis = display.clone();
        cv::putText(vis, info, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);

        cv::imshow(windowName, vis);

        int key = cv::waitKey(0) & 0xFF;
        bool changed = false;

        switch (key) {
            case 27:  // Esc — выход
                cv::destroyAllWindows();
                cv::imwrite("out_drce_loc.png", display);
                return 0;

            case 115: // 's' — сохранить результат
                cv::imwrite("out_drce_loc.png", display);
                std::cout << "Сохранено: out_drce_loc.png\n";
                break;

            case 114: // 'r' — сброс к дефолту
                params.bias     = 100.0;
                params.lambda   = 200.0;
                params.k1       = 1.0;
                params.k2       = 0.4;
                params.dde      = 1.0;
                params.bright   = 128.0;
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
            case ';':   // bright +1  (key right of 'l' on US keyboard)
                params.bright   += 1.0f; changed = true; break;
            case '\'':  // bright -1  (apostrophe, next to ';')
                params.bright   -= 1.0f; changed = true; break;

            default:
                // Непознанная клавиша — игнорируем
                break;
        }

        if (changed) {
            t0 = std::chrono::high_resolution_clock::now();
            DRCELOC algo2(params);
            out = algo2.process(src);
            t1 = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            cv::normalize(out, display, 0, 255, cv::NORM_MINMAX);
            display.convertTo(display, CV_8U);

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