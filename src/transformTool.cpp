//
// Created by mmz on 25-2-18.
//

#include "transformTool.h"

TransformTool::TransformTool() = default;
TransformTool::~TransformTool() = default;

bool TransformTool::getTF(const Corners& corners, const std::vector<Chessboard>& boards,
                          const std::vector<cv::Mat>& chessboards, const CameraParams& cameraParam,
                          std::vector<std::pair<cv::Mat, cv::Mat>>& tf) {
    std::vector<std::vector<cv::Point2f>> points;
    if (!boardClassify(corners, boards, chessboards, points)) {
        std::cout << "boardClassify failed" << std::endl;
        return false;
    }
    if (!pnpCalculate(boards, cameraParam, points, tf)) {
        std::cout << "pnpCalculate failed" << std::endl;
        return false;
    }
    return true;
}

bool TransformTool::getBoardTF(const Corners& corners, const std::vector<Chessboard>& boards,
                               const std::vector<cv::Mat>& chessboards, const CameraParams& cameraParam,
                               std::vector<std::pair<cv::Mat, cv::Mat>>& tf) {
    std::vector<std::pair<cv::Mat, cv::Mat>> TF;
    if (!getTF(corners, boards, chessboards, cameraParam, TF)) {
        std::cout << "getTF failed" << std::endl;
        return false;
    }
    boardTransform(TF, tf);
    return true;
}

bool TransformTool::tfADD(std::pair<cv::Mat, cv::Mat>& transform, std::vector<std::pair<cv::Mat, cv::Mat>>& maskTF,
                          int main, int rest) {
    if (main >= static_cast<int>(maskTF.size())) {
        std::cout << "main out of range" << std::endl;
        return false;
    }
    if (maskTF[main].first.empty() || maskTF[main].second.empty()) {
        std::cout << "maskTF main is empty" << std::endl;
        return false;
    }
    if (main == 0) {
        if (rest > static_cast<int>(maskTF.size())) {
            maskTF.push_back(transform);
        } else {
            if (maskTF[rest].first.empty() || maskTF[rest].second.empty()) {
                maskTF[rest] = transform;
            } else {
                std::cout << "rest is not empty" << std::endl;
                return false;
            }
        }
    } else {
        cv::Mat mainR = maskTF[main].first;
        cv::Mat mainT = maskTF[main].second;
        cv::Mat restR = transform.first;
        cv::Mat restT = transform.second;
        cv::Mat R, T;
        R = mainR.t() * restR;
        T = mainT.t() * (restT - mainT);
        if (rest > static_cast<int>(maskTF.size())) {
            maskTF.emplace_back(R, T);
        } else {
            if (maskTF[rest].first.empty() || maskTF[rest].second.empty()) {
                maskTF[rest] = std::make_pair(R, T);
            } else {
                std::cout << "rest is not empty" << std::endl;
                return false;
            }
        }
    }

    return true;
}


//坐标关系都是相对板1保存，是网状的坐标关系,获得从rest到main的转化关系
bool TransformTool::lookupTransform(const std::vector<std::pair<cv::Mat, cv::Mat>>& maskTF, int main, int rest,
                                    std::pair<cv::Mat, cv::Mat>& transform) {
    if (main >= static_cast<int>(maskTF.size()) || rest >= static_cast<int>(maskTF.size())) {
        std::cout << "main or rest out of range" << std::endl;
        return false;
    }
    if (maskTF[main].first.empty() || maskTF[main].second.empty()) {
        std::cout << "maskTF main is empty" << std::endl;
        return false;
    }
    if (maskTF[rest].first.empty() || maskTF[rest].second.empty()) {
        std::cout << "maskTF rest is empty" << std::endl;
        return false;
    }
    cv::Mat mainR = maskTF[main].first;
    cv::Mat mainT = maskTF[main].second;
    cv::Mat restR = maskTF[rest].first;
    cv::Mat restT = maskTF[rest].second;
    cv::Mat R, T;
    R = mainR.t() * restR;
    // cv::Mat b = restT - mainT;
    T = mainR.t() * (restT - mainT);
    if (cv::norm(T) == 0) {
        T = cv::Mat::zeros(3, 1, mainT.type());
    }

    transform = std::pair<cv::Mat, cv::Mat>(R, T);
    return true;
}


//对单张图片棋盘上的点进行分类
bool TransformTool::boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                                  const std::vector<cv::Mat>& chessboards,
                                  std::vector<std::vector<cv::Point2f>>& points) {
    if (chessboards.empty()) {
        std::cerr << "chessboards is empty" << std::endl;
        return false;
    }
    points.resize(boards.size());
    //遍历所有的棋盘
    for (auto chessboard : chessboards) {
        //对该棋盘进行判断
        for (const auto& board : boards) {
            int crow = chessboard.rows, ccol = chessboard.cols;
            int bx = board.size.first, by = board.size.second;
            if ((ccol == bx || ccol == by) && ccol * crow == bx * by) {
                cv::Mat rcb;
                if (ccol == by) {
                    cv::rotate(chessboard, rcb, cv::ROTATE_90_COUNTERCLOCKWISE);
                } else {
                    rcb = chessboard;
                }
                cv::Point2f left_top_point = corners.p[rcb.at<int>(0, 0)];
                cv::Point2f right_top_point = corners.p[rcb.at<int>(0, ccol - 1)];
                cv::Point2f left_bottom_point = corners.p[rcb.at<int>(crow - 1, 0)];
                cv::Point2f right_bottom_point = corners.p[rcb.at<int>(crow - 1, ccol - 1)];

                std::vector<cv::Point2f> bps{left_top_point, right_top_point, right_bottom_point, left_bottom_point};
                std::sort(bps.begin(), bps.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                    return (a.y + a.x) < (b.y + b.x);
                });

                std::cout << bps[0] << std::endl;

                //判断识别的角度
                if (bps[0]==right_bottom_point) {
                    cv::rotate(rcb, rcb, cv::ROTATE_180);
                }
                if (bps[0]==left_bottom_point) {
                    //垂直翻转
                    cv::flip(rcb, rcb, 0);
                }
                if (bps[0]==right_top_point) {
                    //左右翻转
                    cv::flip(rcb, rcb, 1);
                }
                for (int i = 0; i < rcb.cols; i++) {
                    for (int j = 0; j < rcb.rows; j++) {
                        points[board.id].emplace_back(corners.p[rcb.at<int>(j, i)]);
                    }
                }
            }
        }
    }
    return true;
}

//使用pnp求解该图上每个棋盘格的R T矩阵
bool TransformTool::pnpCalculate(const std::vector<Chessboard>& boards, const CameraParams& cameraParam,
                                 const std::vector<std::vector<cv::Point2f>>& points,
                                 std::vector<std::pair<cv::Mat, cv::Mat>>& TF) {
    if (points.empty()) {
        std::cerr << "points is empty" << std::endl;
        return false;
    }
    TF.resize(boards.size());
    for (int i = 0; i < static_cast<int>(boards.size()); i++) {
        cv::Mat rv, tv, r;
        const auto& board = boards[i];
        const auto& point = points[i];
        if (point.empty()) {
            // r = cv::Mat::eye(3, 3, CV_64F);
            // tv = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
            // TF[i] = std::make_pair(r, tv);
            continue;
        }
        cv::solvePnP(board.objectPoints, point, cameraParam.K, cameraParam.D, rv, tv);

        // std::vector<int> inliers;
        // cv::solvePnPRansac(board.objectPoints, point, cameraParam.K, cameraParam.D, rv, tv, false, // 不使用初始猜测
        //                    300, // 迭代次数
        //                    8.0, // RANSAC重投影误差阈值（像素）
        //                    0.99, // 置信度
        //                    inliers, // 内点索引
        //                    cv::SOLVEPNP_ITERATIVE // 方法类型
        // );
        //
        // if (!inliers.empty()) {
        //     // 提取内点
        //     std::vector<cv::Point3f> inlierObjectPoints;
        //     std::vector<cv::Point2f> inlierImagePoints;
        //     for (int idx : inliers) {
        //         inlierObjectPoints.push_back(board.objectPoints[idx]);
        //         inlierImagePoints.push_back(point[idx]);
        //     }
        //
        //     // 使用初始解进一步优化
        //     cv::solvePnPRefineLM(inlierObjectPoints, inlierImagePoints, cameraParam.K, cameraParam.D, rv, tv);
        // }
        cv::Rodrigues(rv, r);
        if (r.at<double>(2, 2) < 0) {
            // 交换x和y轴
            cv::Mat temp;
            r.col(0).copyTo(temp);
            r.col(1).copyTo(r.col(0));
            temp.copyTo(r.col(1));
            // z轴取负
            r.col(2) = -r.col(2);
        }
        TF[i] = std::make_pair(r, tv);
    }
    return true;
}

void TransformTool::boardTransform(const std::vector<std::pair<cv::Mat, cv::Mat>>& TF,
                                   std::vector<std::pair<cv::Mat, cv::Mat>>& boardTF) {
    if (TF.empty()) {
        std::cerr << "TF is empty" << std::endl;
        return;
    }
    boardTF.resize(TF.size());
    if (TF.size() == 1) {
        boardTF = TF;
        return;
    }
    if (TF[0].first.empty() || TF[0].second.empty()) {
        std::cerr << "TF[0] is empty" << std::endl;
        return;
    }
    cv::Mat r1 = TF[0].first, t1 = TF[0].second;
    cv::Mat r = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tv = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    boardTF[0] = std::make_pair(r, tv);
    for (int i = 1; i < static_cast<int>(TF.size()); i++) {
        cv::Mat R, T;
        if (TF[i].first.empty() || TF[i].second.empty()) {
            continue;
        }
        R = r1.t() * TF[i].first; //相对旋转矩阵
        T = r1.t() * (TF[i].second - t1); //相对平移矩阵
        boardTF[i] = std::make_pair(R, T);
    }
}

void TransformTool::tfAverage(const std::vector<std::pair<cv::Mat, cv::Mat>>& allTF,
                              std::pair<cv::Mat, cv::Mat>& avTF) {
    std::vector<cv::Quatd> q;
    std::vector<cv::Mat> t;
    cv::Mat R, T;
    for (const auto& p : allTF) {
        cv::Mat r = p.first, tv = p.second;
        if (r.empty() || tv.empty()) {
            continue;
        }

        // std::cout << "data" << std::endl;
        // std::cout << r << std::endl;
        // std::cout << tv << std::endl;

        q.emplace_back(cv::Quatd::createFromRotMat(r));
        t.push_back(tv);
    }
    if (q.empty() || t.empty()) {
        std::cout << "q or t is empty" << std::endl;
        return;
    }

    //使用四元数平均R
    {
        int max_iter = 100;
        double epsilon = 1e-8;
    }
    //使用算术平均计算T
    {
        float epsilon = 1e-8f;
        cv::Vec3d sum(0, 0, 0);
        for (const auto& tv : t) {
            sum += cv::Vec3d(tv.at<float>(0), tv.at<float>(1), tv.at<float>(2));
        }
        const int n = static_cast<int>(t.size());
        cv::Vec3d result = sum / n;
        for (int i = 0; i < 3; i++) {
            if (std::abs(result[i]) < epsilon) {
                result[i] = 0.0f;
            }
        }
        T = (cv::Mat_<double>(3, 1) << result[0], result[1], result[2]);
    }
    avTF = std::make_pair(R, T);
}

cv::Vec4f TransformTool::rotationMatrixToQuaternion(const cv::Mat& R) {
    // 确保输入矩阵是3x3的单通道浮点矩阵
    // CV_Assert(R.rows == 3 && R.cols == 3 && R.type() == CV_32F);

    // 提取旋转矩阵的元素
    float m00 = R.at<float>(0, 0), m01 = R.at<float>(0, 1), m02 = R.at<float>(0, 2);
    float m10 = R.at<float>(1, 0), m11 = R.at<float>(1, 1), m12 = R.at<float>(1, 2);
    float m20 = R.at<float>(2, 0), m21 = R.at<float>(2, 1), m22 = R.at<float>(2, 2);

    // 计算矩阵的迹（对角线元素之和）
    float tr = m00 + m11 + m22;
    float qw, qx, qy, qz; // 四元数的四个分量

    // 根据迹的值选择不同的计算方法
    if (constexpr float epsilon = 1e-8f; tr > epsilon) {
        // 迹为正时的计算公式
        float S = sqrt(tr + 1.0f) * 2;
        qw = 0.25f * S;
        qx = (m21 - m12) / S;
        qy = (m02 - m20) / S;
        qz = (m10 - m01) / S;
    } else if (m00 > m11 && m00 > m22) {
        // m00最大时的计算公式
        float S = sqrt(1.0f + m00 - m11 - m22) * 2;
        qw = (m21 - m12) / S;
        qx = 0.25f * S;
        qy = (m01 + m10) / S;
        qz = (m02 + m20) / S;
    } else if (m11 > m22) {
        // m11最大时的计算公式
        float S = sqrt(1.0f + m11 - m00 - m22) * 2;
        qw = (m02 - m20) / S;
        qx = (m01 + m10) / S;
        qy = 0.25f * S;
        qz = (m12 + m21) / S;
    } else {
        // m22最大时的计算公式
        float S = sqrt(1.0f + m22 - m00 - m11) * 2;
        qw = (m10 - m01) / S;
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25f * S;
    }

    // 归一化处理
    const cv::Vec4f q(qw, qx, qy, qz);
    return q / cv::norm(q);
}

// 统一四元数的符号
void TransformTool::unifyQuaternionSigns(std::vector<cv::Vec4f>& quaternions) {
    if (quaternions.empty()) return;
    cv::Vec4f base = quaternions[0];
    for (auto& q : quaternions) {
        if (base.dot(q) < 0) q *= -1;
    }
}

// 将四元数转换为旋转矩阵
cv::Mat TransformTool::quaternionToRotationMatrix(const cv::Vec4f& q) {
    float epsilon = 1e-8f;

    float w = q[0], x = q[1], y = q[2], z = q[3];

    // 计算矩阵元素
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    cv::Mat R = (cv::Mat_<float>(3, 3) << 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), 1 - 2 * (xx +
        zz), 2 * (yz - wx), 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));

    // 数值截断：清除接近零的值
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            auto& val = R.at<float>(i, j);
            if (std::abs(val) < epsilon) val = 0.0f;
        }
    }

    // 强制正交化（通过SVD）
    cv::Mat U, S, Vt;
    cv::SVDecomp(R, S, U, Vt);
    R = U * Vt;

    return R;
}
