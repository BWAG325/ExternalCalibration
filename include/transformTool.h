//
// Created by mmz on 25-2-18.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "CornerDetAC.h"
#include "ChessboradStruct.h"


class TransformTool {
    //对现检测到的棋盘格按照行列进行分类，储存到points中
    void boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                       const std::vector<cv::Mat>& chessboards, std::vector<std::vector<cv::Point2f>> points);

    void PNPcalculate(const std::vector<Chessboard>& boards, const std::vector<std::vector<cv::Point2f>> points,
                      std::vector<std::vector<std::pair<cv::Mat,cv::Mat>>>& boadTF);
public:
    TransformTool();
    ~TransformTool();

    //TODO 支持棋盘格变化查找功能
    bool getTransform(const std::vector<std::pair<cv::Mat,cv::Mat>>& boardTf);
};

