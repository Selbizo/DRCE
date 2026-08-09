// Демо/тест DRCE-LOC.
#include <iomanip>
#include <vector>
#include <utility>
//
//
// Если путь не задан или файл не открылся — генерируется синтетическая
// сцена с высоким динамическим диапазоном (аналог "rich scene" из статьи:
// яркое небо + тёмное здание + мелкие цели), сохранённая как 16-битный
// PNG, чтобы было на чём проверить сжатие ДД 14/16 бит -> 8 бит.


#include "drce_loc.hpp"
#include <chrono>
#include <iostream>

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

    // --- DRCE-LOC ---
    DRCELOC::Params params;
    // Размер блока ~64x64 при 1024x768, как рекомендует Section 2.4 статьи.
    params.blockRows = std::max(1, src.rows / 32);
    params.blockCols = std::max(1, src.cols / 32);
    params.delta2   = -1.0;
    params.bias   = 100.0;
    params.lambda = 200.0;
    params.k1 = 1.0;
    params.k2 = 0.4;
    params.dde = 1.0;
    params.bright = 128.0;

    DRCELOC algo(params);

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat out = algo.process(src);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "DRCE-LOC время обработки: " << ms << " мс "
              << "(blockRows=" << params.blockRows << ", blockCols=" << params.blockCols << ")\n";

    cv::imwrite("out_drce_loc.png", out);

    // Промежуточные слои — для отладки/апробации (сравнение с Figure 4-5 статьи).
    // Нумерация соответствует шагам алгоритма (Section 2.1, Figure 2 статьи).
    saveNorm("step01_block_std.png",  algo.debug().blockStd);
    saveNorm("step03_NStretch.png",   algo.debug().NStretch);
    saveNorm("step03_NMean.png",      algo.debug().NMean);
    saveNorm("step04_gf_detail.png",  algo.debug().gfDetail);
    saveNorm("step05_Igc.png",        algo.debug().Igc);
    saveNorm("step06_base_out.png",   algo.debug().baseOut);
    saveNorm("step07_detail_out.png", algo.debug().detailOut);

    // --- Метрики из статьи (Section 3.1): RMS (Eq.9), Entropy (Eq.10), Tenengrad (Eq.13) ---
    auto computeRMS = [](const cv::Mat& img8u) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(img8u, mean, stddev);
        return stddev[0]; // RMS относительно среднего, как в Eq. 9
    };
    auto computeEntropy = [](const cv::Mat& img8u) {
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&img8u, 1, 0, cv::Mat(), hist, 1, &histSize, ranges);
        hist /= (double)(img8u.rows * img8u.cols);
        double entropy = 0.0;
        for (int i = 0; i < histSize; ++i) {
            float p = hist.at<float>(i);
            if (p > 1e-9f) entropy -= p * std::log2(p);
        }
        return entropy;
    };
    auto computeTenengrad = [](const cv::Mat& img8u) {
        cv::Mat gx, gy;
        cv::Sobel(img8u, gx, CV_32F, 1, 0, 3);
        cv::Sobel(img8u, gy, CV_32F, 0, 1, 3);
        cv::Mat mag2 = gx.mul(gx) + gy.mul(gy);
        return cv::mean(mag2)[0];
    };

    std::cout << "\n--- Сравнение метрик (Section 3.1 статьи) ---\n";
    std::cout << "                RMS      Entropy   Tenengrad\n";
    for (auto& [name, img] : std::vector<std::pair<std::string, cv::Mat>>{
             {"Linear/AGC", baseline8u}, {"CLAHE", claheOut}, {"DRCE-LOC", out}}) {
        std::cout << std::left << std::setw(14) << name
                  << std::right << std::setw(8) << computeRMS(img)
                  << std::setw(11) << computeEntropy(img)
                  << std::setw(12) << computeTenengrad(img) << "\n";
    }

    std::cout << "\nФайлы сохранены в текущей директории "
              << "(out_*.png — результаты, stepNN_*.png — промежуточные слои по шагам 1-8).\n";
    return 0;
}