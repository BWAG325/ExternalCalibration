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


//对单张图片棋盘上的点进行分类
bool TransformTool::boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                                  const std::vector<cv::Mat>& chessboards,
                                  std::vector<std::vector<cv::Point2f>> points) {
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
                for (int i = 0; i < chessboard.rows; i++) {
                    for (int j = 0; j < chessboard.cols; j++) {
                        points[board.id].push_back(corners.p[chessboard.at<int>(i, j)]);
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
    for (int i = 0; i < boards.size(); i++) {
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

void TransformTool::boardTransform(const std::vector<std::pair<cv::Mat, cv::Mat>>& TF, std::vector<std::pair<cv::Mat, cv::Mat>>& boardTF) {
    boardTF.resize(TF.size());
    if (TF.size() == 1) {
        boardTF = TF;
        return;
    }
    cv::Mat r1 = TF[0].first, t1 = TF[0].second;
    cv::Mat r = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tv = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    boardTF[0] = std::make_pair(r, tv);
    for (int i = 1; i < TF.size(); i++) {
        cv::Mat R,T;
        if (TF[i].first.empty() && TF[i].second.empty()) {
            continue;
        }
        R = r1.t() * TF[i].first;//相对旋转矩阵
        T = r1.t() * (TF[i].second - t1);//相对平移矩阵
        boardTF[i] = std::make_pair(R, T);
    }
}
