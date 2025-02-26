//
// Created by mmz on 25-2-15.
//

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <transformTool.h>

#include "CornerDetAC.h"
#include "ChessboradStruct.h"

void showTF(std::vector<std::pair<cv::Mat, cv::Mat>> boradsTF) {
    for (auto it : boradsTF) {
        std::cout << "RT" << std::endl;
        std::cout << it.first << std::endl;
        std::cout << it.second << std::endl;
    }
}

std::string yamlPath = "../data/params.yaml";

//读取参数
void loadParams(std::vector<CameraParams>& cameraParams, std::vector<Chessboard>& boards) {
    std::cout << "Loading ..." << std::endl;
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error opening file " << yamlPath << std::endl;
    }
    cv::FileNode boardNode = fs["boards"];
    for (auto&& it : boardNode) {
        Chessboard board(it["id"], std::pair<int, int>(it["X"], it["Y"]), it["square_size"]);
        boards.push_back(board);
    }
    cv::FileNode cameraNode = fs["cameras"];
    for (auto&& it : cameraNode) {
        CameraParams cam;
        it["id"] >> cam.id;
        it["K"] >> cam.K;
        it["D"] >> cam.D;
        cameraParams.push_back(cam);
    }
}

//读取图片
cv::Mat getImage(const std::string& name, const std::string& type) {
    std::string img_path = "../img/" + name + "." + type;
    std::cout << "reading img" << img_path << std::endl;
    cv::Mat src = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (src.empty()) std::cout << "empty image" << std::endl;
    //转化成灰度图
    cv::Mat gray;
    switch (src.channels()) {
    case 1: gray = src.clone();
        break;
    case 3: cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        break;
    case 4: cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        break;
    default:
        std::cout << "unsupported image format" << std::endl;
    }
    std::cout << "read image succeed" << std::endl;

    return gray;
}

//检测棋盘格
void findBoard(cv::Mat& gray, Corners& corners, std::vector<cv::Mat>& chessboards) {
    //START
    CornerDetAC cornerDetector;
    ChessboardStruct chessboardStruct;
    auto timeCount = static_cast<double>(cv::getTickCount());
    cornerDetector.detectCorners(gray, corners, 0.01);
    chessboardStruct.chessboardsFromCorners(corners, chessboards, 0.8);
    timeCount = (static_cast<double>(cv::getTickCount()) - timeCount) / cv::getTickFrequency();
    std::cout << "time cost :" << timeCount << std::endl;
    //END
    // chessboardStruct.drawChessboard(gray, corners, chessboards, "board");
}

//NOTE remove
bool boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                   const std::vector<cv::Mat>& chessboards, std::vector<std::vector<cv::Point2f>>& points) {
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
                    } else {
                        for (int i = 0; i < crow; i++) {
                            for (int j = 0; j < ccol; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(crow - i - 1, ccol - j - 1)]);
                            }
                        }
                    }
                } else {
                    if (tp.y < bp.y) {
                        for (int i = 0; i < ccol; i++) {
                            for (int j = 0; j < crow; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(j, i)]);
                            }
                        }
                    } else {
                        for (int i = 0; i < ccol; i++) {
                            for (int j = 0; j < crow; j++) {
                                points[board.id].push_back(corners.p[chessboard.at<int>(crow - j - 1, ccol - 1 - i)]);
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

void singleBoardImg(cv::Mat& gray, const std::vector<Chessboard>& boards, const CameraParams& cameraParam,
                    std::vector<std::pair<cv::Mat, cv::Mat>>& tf) {
    std::vector<cv::Mat> chessboards;
    Corners corners;
    findBoard(gray, corners, chessboards);
    //NOTE remove
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2RGB);
    std::vector<std::vector<cv::Point2f>> points;
    boardClassify(corners, boards, chessboards, points);
    for (int i = 0; i < points.size(); i++) {
        cv::drawChessboardCorners(gray, cv::Size(boards[i].size.first, boards[i].size.second), points[i], true);
    }

    TransformTool::getTF(corners, boards, chessboards, cameraParam, tf);
    //show NOTE remove
    for (auto&& it : tf) {
        if (it.first.empty() || it.second.empty()) {
            continue;
        }
        cv::Mat R = it.first;
        cv::drawFrameAxes(gray, cameraParam.K, cameraParam.D, R, it.second, 0.5);
        cv::imshow("b", gray);
        cv::waitKey(0);
    }
};

void getBoardTF(const std::string& name, const int num, const std::string& type, const std::vector<Chessboard>& boards,
                const CameraParams& cameraParam, std::vector<std::pair<cv::Mat, cv::Mat>>& tf) {
    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> TFs;
    TFs.resize(boards.size());
    for (int i = 0; i < num; i++) {
        std::vector<std::pair<cv::Mat, cv::Mat>> boardTF;
        std::string n = name + "" + std::to_string(i);
        cv::Mat gray = getImage(n, type);
        singleBoardImg(gray, boards, cameraParam, boardTF);
        for (int j = 0; j < boardTF.size(); j++) {
            TFs[j].push_back(boardTF[j]);
        }
        tf = boardTF;
    }
    for (auto it : TFs) {
        std::pair<cv::Mat, cv::Mat> avTF;
        TransformTool::tfAverage(it, avTF);
        tf.push_back(avTF);
    }
    showTF(tf);
    std::cout << "show board" << std::endl;
}


int main() {
    //加载相关相机内参以及标定版参数
    std::vector<Chessboard> boards;
    std::vector<CameraParams> cameraParams;
    loadParams(cameraParams, boards);

    std::vector<std::pair<cv::Mat, cv::Mat>> boradsTF;
    getBoardTF("t", 4, "png", boards, cameraParams[0], boradsTF);

    std::pair<cv::Mat, cv::Mat> a;
    TransformTool::lookupTransform(boradsTF, 0, 3, a);
    std::cout << "a" << std::endl;
    std::cout << a.first << std::endl;
    std::cout << a.second << std::endl;

    // cv::Mat rvec, tvec;  // 旋转向量和平移向量
    // cv::solvePnP(objectPoints, imgPoints, K, D, rvec, tvec);


    // chessboardStruct.drawChessboard(src, corners, chessboards, "board");

    return 0;
}
