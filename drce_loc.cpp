#include "drce_loc.hpp"

// ----------------------------------------------------------------------------
// Step 1: Image segmentation + Block_mean / Block_std
// ----------------------------------------------------------------------------
void DRCELOC::computeBlockStats(const cv::Mat& img32f, cv::Mat& blockMean, cv::Mat& blockStd) {
    const int M = img32f.rows, N = img32f.cols;
    const int X = params_.blockRows, Y = params_.blockCols;

    blockMean.create(X, Y, CV_64F);
    blockStd.create(X, Y, CV_64F);

    for (int bx = 0; bx < X; ++bx) {
        int r0 = static_cast<int>(std::round(bx * (double)M / X));
        int r1 = static_cast<int>(std::round((bx + 1) * (double)M / X));
        r1 = std::max(r1, r0 + 1);
        for (int by = 0; by < Y; ++by) {
            int c0 = static_cast<int>(std::round(by * (double)N / Y));
            int c1 = static_cast<int>(std::round((by + 1) * (double)N / Y));
            c1 = std::max(c1, c0 + 1);

            cv::Rect roi(c0, r0, c1 - c0, r1 - r0);
            cv::Scalar mean, stddev;
            cv::meanStdDev(img32f(roi), mean, stddev);

            blockMean.at<double>(bx, by) = mean[0];
            blockStd.at<double>(bx, by)  = stddev[0];
        }
    }
}

// ----------------------------------------------------------------------------
// Step 3: Gaussian upsampling блочной карты (X x Y) до полного разрешения
// (M x N), Eq. 2 / Eq. 3.
//
// Точная формула:
//   N(i,j) = sum_{x,y} V(x,y)*exp(-d((i,j),(x,y))^2/delta2) / sum_{x,y} exp(-d^2/delta2)
//
// Разделимость (см. комментарий в drce_loc.hpp и Eq. 15 статьи):
//   exp(-d^2/delta2) = exp(-dRow(i,x)^2/delta2) * exp(-dCol(j,y)^2/delta2)
//
// Тогда числитель = Wrow (MxX) * V (XxY) * Wcol^T (YxN)  -> M x N,
//      знаменатель = rowSum(Wrow) (Mx1) * colSum(Wcol)^T (1xN) -> M x N (outer product),
// что считается двумя cv::gemm вместо O(M*N*X*Y) перебора.
// ----------------------------------------------------------------------------
cv::Mat DRCELOC::gaussianUpsample(const cv::Mat& blockMap64f, int M, int N, double delta2) {
    const int X = blockMap64f.rows, Y = blockMap64f.cols;

    // Центры блоков в координатах полного изображения.
    std::vector<double> rowCenters(X), colCenters(Y);
    for (int bx = 0; bx < X; ++bx) rowCenters[bx] = (bx + 0.5) * (double)M / X;
    for (int by = 0; by < Y; ++by) colCenters[by] = (by + 0.5) * (double)N / Y;

    cv::Mat Wrow(M, X, CV_64F);   // Wrow(i,x) = exp(-(i-cx)^2/delta2)
    cv::Mat Wcol(N, Y, CV_64F);   // Wcol(j,y) = exp(-(j-cy)^2/delta2)

    for (int i = 0; i < M; ++i)
        for (int bx = 0; bx < X; ++bx) {
            double d = i - rowCenters[bx];
            Wrow.at<double>(i, bx) = std::exp(-(d * d) / delta2);
        }
    for (int j = 0; j < N; ++j)
        for (int by = 0; by < Y; ++by) {
            double d = j - colCenters[by];
            Wcol.at<double>(j, by) = std::exp(-(d * d) / delta2);
        }

    // Числитель: (Wrow * V) * Wcol^T
    cv::Mat temp = Wrow * blockMap64f;      // M x Y
    cv::Mat numerator = temp * Wcol.t();    // M x N

    // Знаменатель: rowSum outer colSum
    cv::Mat rowSum, colSum;
    cv::reduce(Wrow, rowSum, 1, cv::REDUCE_SUM); // M x 1
    cv::reduce(Wcol, colSum, 1, cv::REDUCE_SUM); // N x 1
    cv::Mat denom = rowSum * colSum.t();         // M x N (outer product)

    cv::Mat result;
    cv::divide(numerator, denom, result);
    result.convertTo(result, CV_32F);
    return result;
}

// ----------------------------------------------------------------------------
// Step 4: Self-guided filter (p = I, guide = I), окно w x w, Eq. 4.
// Стандартная реализация через box filter (He et al., 2012), специализация
// для self-guided случая: a_k = var_I/(var_I+eps); b_k = mean_I*(1-a_k).
// ----------------------------------------------------------------------------
void DRCELOC::selfGuidedFilter(const cv::Mat& img32f, cv::Mat& base, cv::Mat& detail, cv::Mat& aCoeff) {
    const int r = params_.guidedRadius;
    const cv::Size ksize(2 * r + 1, 2 * r + 1);
    const double eps = params_.guidedEps;

    cv::Mat meanI, meanII, varI;
    cv::boxFilter(img32f, meanI, CV_32F, ksize);
    cv::boxFilter(img32f.mul(img32f), meanII, CV_32F, ksize);
    varI = meanII - meanI.mul(meanI);

    cv::Mat a = varI / (varI + eps);
    cv::Mat b = meanI - a.mul(meanI); // = meanI*(1-a)

    cv::Mat meanA, meanB;
    cv::boxFilter(a, meanA, CV_32F, ksize);
    cv::boxFilter(b, meanB, CV_32F, ksize);

    base = meanA.mul(img32f) + meanB;
    detail = img32f - base;
    aCoeff = a;
}

// ----------------------------------------------------------------------------
// Step 5: Global brightness guide map, Eq. 5.
// ----------------------------------------------------------------------------
cv::Mat DRCELOC::computeBrightnessGuide(const cv::Mat& img32f) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img32f, mean, stddev);

    cv::Mat Igc = (255.0 * (img32f - mean[0])) / (stddev[0] + params_.lambda) + params_.bright;
    return Igc;
}

// ----------------------------------------------------------------------------
// Основной конвейер: Steps 1-8.
// ----------------------------------------------------------------------------
cv::Mat DRCELOC::process(const cv::Mat& src) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 1);

    cv::Mat img32f;
    src.convertTo(img32f, CV_32F);

    if (params_.normalizeInputTo255) {
        double lo, hi;
        cv::minMaxLoc(img32f, &lo, &hi);
        if (hi - lo > 1e-6) {
            img32f = (img32f - lo) * (255.0 / (hi - lo));
        }
    }

    const int M = img32f.rows, N = img32f.cols;
    const int X = params_.blockRows, Y = params_.blockCols;

    // --- Step 1-2 ---
    computeBlockStats(img32f, dbg_.blockMean, dbg_.blockStd);
    cv::Mat blockStretch = 255.0 / (dbg_.blockStd + params_.bias); // Eq. 1
    dbg_.blockStretch = blockStretch;

    // --- Step 3 ---
    double delta2 = params_.delta2;
    if (delta2 <= 0.0) {
        double blockArea = ((double)M / X) * ((double)N / Y);
        delta2 = 1.5 * blockArea; // статья: delta^2 >= размер блока (Section 2.4)
    }
    dbg_.NStretch = gaussianUpsample(blockStretch, M, N, delta2);
    dbg_.NMean    = gaussianUpsample(dbg_.blockMean, M, N, delta2);

    // --- Step 4 ---
    selfGuidedFilter(img32f, dbg_.gfBase, dbg_.gfDetail, dbg_.gfA);

    // --- Step 5 ---
    dbg_.Igc = computeBrightnessGuide(img32f);

    // --- Step 6: Eq. 6 ---
    // Base_out = k1*NStretch*(Iin - NMean) + k2*Igc
    cv::Mat baseOut = params_.k1 * dbg_.NStretch.mul(img32f - dbg_.NMean) + params_.k2 * dbg_.Igc;
    dbg_.baseOut = baseOut;

    // --- Step 7: Eq. 7 (интерпретация, см. комментарий в drce_loc.hpp) ---
    cv::Mat absA = cv::abs(dbg_.gfA);
    double maxA;
    cv::minMaxLoc(absA, nullptr, &maxA);
    if (maxA < 1e-9) maxA = 1.0;
    cv::Mat normA = absA / maxA; // 0..1, мера локальной "детализированности"
    cv::Mat mask = params_.gLow + (params_.gHigh - params_.gLow) * normA;
    cv::Mat detailOut = dbg_.gfDetail.mul(absA).mul(mask);
    dbg_.detailOut = detailOut;

    // --- Step 8: Eq. 8 ---
    cv::Mat out32f = baseOut + params_.dde * detailOut;

    cv::Mat out8u;
    cv::Mat clipped;
    cv::min(cv::max(out32f, 0.0), 255.0, clipped);
    clipped.convertTo(out8u, CV_8U);
    return out8u;
}
