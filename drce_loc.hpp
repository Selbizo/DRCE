// ============================================================================
// DRCE-LOC: Dynamic Range Compression and Contrast Enhancement algorithm
// with Local Optimal Contrast.
//
// Реализация по статье:
//   Zhu Y., Zhou Y., Jin W., Zhang L., Wu G., Shao Y.
//   "A Low-Delay Dynamic Range Compression and Contrast Enhancement Algorithm
//   Based on an Uncooled Infrared Sensor with Local Optimal Contrast"
//   Sensors 2023, 23, 8860. https://doi.org/10.3390/s23218860
//
// Соответствие шагам статьи (Section 2.1, Figure 2):
//   Step 1  computeBlockStats()      -> Block_mean, Block_std      (Eq. -)
//   Step 2  computeStretchCoeff()    -> Stretch_para                (Eq. 1)
//   Step 3  gaussianUpsample()       -> NStretch, NMean             (Eq. 2,3)
//   Step 4  selfGuidedFilter()       -> Base(GF), Detail, a_coeff   (Eq. 4)
//   Step 5  computeBrightnessGuide() -> Igc                         (Eq. 5)
//   Step 6  process(): Base_out                                    (Eq. 6)
//   Step 7  process(): Detail_out (noise mask)                     (Eq. 7)
//   Step 8  process(): Iout = Base_out + DDE*Detail_out             (Eq. 8)
//
// Примечание по шагу 3 (Gaussian upsampling, Eq. 2-3):
//   Ядро exp(-d(i,j)(x,y)^2 / delta^2) разделимо по строкам/столбцам —
//   ровно та же идея, что авторы используют в аппаратной реализации
//   (Section 3.2, Part 5, Eq. 15: exp(-d^2/delta^2) = ex * ey).
//   Здесь это используется не для экономии LUT на FPGA, а для замены
//   O(M*N*X*Y) наивного апсемплинга на два матричных умножения (cv::gemm),
//   что даёт ТОЧНОЕ (не аппроксимированное) значение формулы Eq. 2-3 за
//   O(M*N*(X+Y)) операций.
//
// Примечание по шагу 7 (шумовая маска детального слоя, Eq. 7):
//   В статье формула Detail_out = Detail*|a_k|*[gL+(gH-gL)] ссылается на
//   внешний источник (ref. [17], Zuo et al. 2011) для полного вида функции
//   и не специфицирует её аргумент внутри статьи. Здесь реализована
//   типовая интерпретация принципа "глаз чувствителен к шуму в однородных
//   зонах и нечувствителен в детализированных": весовая маска строится по
//   нормированному коэффициенту a_k управляемого фильтра (мера локальной
//   информативности/дисперсии), gL/gH — параметры алгоритма (см. Params).
//   Это единственное место в реализации, где пришлось доопределить
//   недостающую в статье деталь; при необходимости легко заменить на
//   собственную функцию маски.
// ============================================================================

#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

class DRCELOC {
public:
    struct Params {
        // Предварительная нормализация входа в номинальный диапазон 0..255
        // (min-max по кадру) до блочной статистики. Нужна потому, что Bias/
        // lambda в статье (128-384) заданы для этого диапазона (см. Eq. 5,
        // "8-bit global brightness guide"); без неё параметры, подобранные
        // авторами, не переносятся на сырые 14/16-битные данные "как есть".
        bool normalizeInputTo255 = true;

        // --- Шаг 1-3: блочная статистика и её апсемплинг ---
        int blockRows = 12;     // X: число блоков по вертикали
        int blockCols = 16;     // Y: число блоков по горизонтали
                                 // (12x16 при 1024x768 -> блок ~64x64, как в статье, Section 2.4)
        double bias = 200.0;    // Bias, Eq. 1. Статья: 128-384 (подавление
                                 // локального переусиления шума, Section 2.2)
        double delta2 = -1.0;   // delta^2, Eq. 2-3. Статья: delta^2 >= (M/X)*(N/Y).
                                 // При delta2 <= 0 выбирается автоматически как
                                 // 1.5 * (размер блока по площади).

        // --- Шаг 4: guided filter (self-guided, окно 5x5 -> radius=2) ---
        int guidedRadius = 2;
        double guidedEps = 100.0;   // регуляризация в Eq. 4 (epsilon)

        // --- Шаг 5: глобальная яркостная направляющая карта ---
        double lambda = 200.0;  // lambda, Eq. 5. Статья: 128-384
        double bright = 128.0;  // Bright, Eq. 5 — желаемая средняя яркость

        // --- Шаг 6: компрессия ДД + контрастное усиление ---
        double k1 = 1.0;        // вес локального контраста, Eq. 6 (статья: k1=1)
        double k2 = 0.4;        // вес глобального контраста / подавление гало,
                                 // Eq. 6. Статья: диапазон 0.2-0.7 (Section 2.3)

        // --- Шаг 7: шумовая маска детального слоя ---
        double gLow  = 0.3;     // минимальный gain в однородных зонах
        double gHigh = 1.2;     // максимальный gain в детализированных зонах

        // --- Шаг 8: финальный синтез ---
        double dde = 1.0;       // Detail enhancement factor, Eq. 8
    };

    DRCELOC() : params_() {}
    explicit DRCELOC(const Params& p) : params_(p) {}

    // src: одноканальное изображение любой глубины (CV_8U/CV_16U/CV_32F).
    // Возвращает CV_8U изображение того же размера.
    cv::Mat process(const cv::Mat& src);

    // Доступ к промежуточным слоям последнего вызова process() — удобно
    // для апробации/отладки (посмотреть Base_out, Detail_out, Igc и т.д.
    // отдельно, как на Figure 4-5 статьи).
    struct DebugLayers {
        cv::Mat blockMean, blockStd, blockStretch; // X x Y карты (шаг 1-2)
        cv::Mat NMean, NStretch;                   // M x N, после апсемплинга (шаг 3)
        cv::Mat gfBase, gfDetail, gfA;              // выход guided filter (шаг 4)
        cv::Mat Igc;                                 // яркостная направляющая (шаг 5)
        cv::Mat baseOut;                             // Eq. 6
        cv::Mat detailOut;                           // Eq. 7
    };
    const DebugLayers& debug() const { return dbg_; }

private:
    Params params_;
    DebugLayers dbg_;

    void computeBlockStats(const cv::Mat& img32f, cv::Mat& blockMean, cv::Mat& blockStd);
    cv::Mat gaussianUpsample(const cv::Mat& blockMap, int M, int N, double delta2);
    void selfGuidedFilter(const cv::Mat& img32f, cv::Mat& base, cv::Mat& detail, cv::Mat& aCoeff);
    cv::Mat computeBrightnessGuide(const cv::Mat& img32f);
};
