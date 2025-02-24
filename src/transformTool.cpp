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
    std::cout << mainR << std::endl;
    std::cout << mainT << std::endl;
    std::cout << restR << std::endl;
    std::cout << restT << std::endl;
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
            int ccol = chessboard.cols, crow = chessboard.rows;
            int bx = board.size.first, by = board.size.second;
            if ((ccol == bx || ccol == by) && ccol * crow == bx * by) {
                //遍历改棋盘的每一个点,重新储存
                int top = chessboard.at<int>(0, 0);
                int bottom = chessboard.at<int>(crow - 1, ccol - 1);
                cv::Point2f tp = corners.p[top];
                cv::Point2f bp = corners.p[bottom];
                if (ccol == bx) {
                    if (tp.y < bp.y) {
                        for (int i = 0; i < crow; i++) {
                            for (int j = 0; j < ccol; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(i, j)]);
                            }
                        }
                    }else {
                        for (int i = 0; i < crow; i++) {
                            for (int j = 0; j < ccol; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(crow-i-1, ccol-j-1)]);
                            }
                        }
                    }
                }else {
                    if (tp.y < bp.y) {
                        for (int i = 0; i < ccol; i++) {
                            for (int j = 0; j < crow; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(j, i)]);
                            }
                        }
                    }else {
                        for (int i = 0; i < ccol; i++) {
                            for (int j = 0; j < crow; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(crow - j - 1, ccol-1-i)]);
                            }
                        }
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
        cv::Rodrigues(rv, r);
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
    std::vector<cv::Vec4f> q;
    std::vector<cv::Mat> t;
    cv::Mat R, T;
    for (const auto& p : allTF) {
        cv::Mat r = p.first, tv = p.second;
        if (r.empty() || tv.empty()) {
            continue;
        }
        q.push_back(rotationMatrixToQuaternion(r));
        t.push_back(tv);
    }
    if (q.empty()||t.empty()) {
        std::cout << "q or t is empty" << std::endl;
        return;
    }
    unifyQuaternionSigns(q);
    //使用四元数平均R
    {
        cv::Vec4f sum(0, 0, 0, 0); // 初始化四元数和为零
        for (const auto& quaternion : q) {
            sum += quaternion;
        } // 累加所有四元数
        sum /= static_cast<float>(q.size()); // 计算平均值
        cv::Vec4f Q = sum / cv::norm(sum); // 归一化平均四元数并返回
        R = quaternionToRotationMatrix(Q);
    }
    //使用算术平均计算T
    {
        cv::Vec3f sum(0, 0, 0);
        for (const auto& tv : t) {
            sum += cv::Vec3f(tv.at<float>(0), tv.at<float>(1), tv.at<float>(2));
        }
        const int n = static_cast<int>(t.size());
        cv::Vec3f result = sum / n;
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
    if (tr > 0) {
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

    // 返回四元数（qw, qx, qy, qz）
    return {qw, qx, qy, qz};
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
    float w = q[0], x = q[1], y = q[2], z = q[3]; // 提取四元数的分量
    float xx = x * x, xy = x * y, xz = x * z, xw = x * w;
    float yy = y * y, yz = y * z, yw = y * w;
    float zz = z * z, zw = z * w;

    // 根据四元数的公式构造旋转矩阵
    return (cv::Mat_<float>(3, 3) << 1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw, 2 * xy + 2 * zw, 1 - 2 * xx
        - 2 * zz, 2 * yz - 2 * xw, 2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy);
}
