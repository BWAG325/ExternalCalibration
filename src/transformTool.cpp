//
// Created by mmz on 25-2-18.
//

#include "transformTool.h"

TransformTool::TransformTool() = default;
TransformTool::~TransformTool() = default;

//对棋盘上的点进行分类
void TransformTool::boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                            const std::vector<cv::Mat>& chessboards, std::vector<std::vector<cv::Point2f>> points) {
    points.resize(boards.size());
    //遍历所有的棋盘
    for (auto chessboard : chessboards) {
        //对该棋盘进行判断
        for (auto board : boards) {
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
}

//使用pnp求解每张图上每个棋盘格的R T矩阵
void TransformTool::PNPcalculate(const std::vector<Chessboard>& boards, const std::vector<std::vector<cv::Point2f>> points,
                  std::vector<std::vector<std::pair<cv::Mat,cv::Mat>>>& boadTF){
  boadTF.resize(boards.size());
                  }