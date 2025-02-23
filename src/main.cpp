//
// Created by mmz on 25-2-15.
//

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <transformTool.h>

#include "CornerDetAC.h"
#include "ChessboradStruct.h"

CornerDetAC cornerDetector;
ChessboardStruct chessboardStruct;
std::string yamlPath = "../data/params.yaml";

//读取参数
void loadParams(std::vector<CameraParams>& cameraParams, std::vector<Chessboard>& boards) {
    std::cout << "Loading ..." << std::endl;
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error opening file " << yamlPath << std::endl;
    }
    cv::FileNode boardNode = fs["boards"];
    for (cv::FileNodeIterator it = boardNode.begin(); it != boardNode.end(); ++it) {
        Chessboard board((*it)["id"], std::pair<int, int>((*it)["X"], (*it)["Y"]), (*it)["square_size"]);
        boards.push_back(board);
    }
    cv::FileNode cameraNode = fs["cameras"];
    for (cv::FileNodeIterator it = cameraNode.begin(); it != cameraNode.end(); ++it) {
        CameraParams cam;
        (*it)["id"] >> cam.id;
        (*it)["K"] >> cam.K;
        (*it)["D"] >> cam.D;
        cameraParams.push_back(cam);
    }
}

//读取图片
cv::Mat getImage(cv::Mat& src) {
    std::string img_path = "../img/5.png";
    std::cout << "reading img" << img_path << std::endl;
    src = cv::imread(img_path, cv::IMREAD_UNCHANGED);
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
    auto timeCount = static_cast<double>(cv::getTickCount());
    cornerDetector.detectCorners(gray, corners, 0.01);
    chessboardStruct.chessboardsFromCorners(corners, chessboards, 0.8);
    timeCount = (static_cast<double>(cv::getTickCount()) - timeCount) / cv::getTickFrequency();
    std::cout << "time cost :" << timeCount << std::endl;
    //END
}

void singleImg(cv::Mat& gray,const std::vector<Chessboard>& boards,const CameraParams& cameraParam,
          std::vector<std::pair<cv::Mat, cv::Mat>>& tf) {
    std::vector<cv::Mat> chessboards;
    Corners corners;
    findBoard(gray, corners, chessboards);
    TransformTool::getBoardTF(corners,boards,chessboards,cameraParam,tf);
};

int main() {
    //加载相关相机内参以及标定版参数
    std::vector<Chessboard> boards;
    std::vector<CameraParams> cameraParams;
    loadParams(cameraParams, boards);

    cv::Mat src;
    cv::Mat gray = getImage(src);


    std::vector<std::pair<cv::Mat, cv::Mat>> boradsTF;
    singleImg(gray,boards,cameraParams[0],boradsTF);
    std::cout << boards.size() << std::endl;

    std::pair<cv::Mat, cv::Mat> a;
    TransformTool::lookupTransform(boradsTF,0,0,a);
    std::cout << a.first << std::endl;
    std::cout << a.second << std::endl;

    // cv::Mat rvec, tvec;  // 旋转向量和平移向量
    // cv::solvePnP(objectPoints, imgPoints, K, D, rvec, tvec);


    // chessboardStruct.drawChessboard(src, corners, chessboards, "board");

    return 0;
}



