//
// Created by mmz on 25-2-18.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <vector>

#include "CornerDetAC.h"
#include "ChessboradStruct.h"


class TransformTool {
    //单张图片现检测到的棋盘格按照行列进行分类，储存到points中
    static bool boardClassify(const Corners& corners, const std::vector<Chessboard>& boards,
                                     const std::vector<cv::Mat>& chessboards,
                                     std::vector<std::vector<cv::Point2f>>& points);
    //对该图片上的棋盘格进行pnp，储存到TF中
    static bool pnpCalculate(const std::vector<Chessboard>& boards, const CameraParams& cameraParam,
                             const std::vector<std::vector<cv::Point2f>>& points,
                             std::vector<std::pair<cv::Mat, cv::Mat>>& TF);


    //对该图片上的棋盘格变化关系进行求解,储存到boardTF中,全部转化成n到0的变化矩阵
    static void boardTransform(const std::vector<std::pair<cv::Mat, cv::Mat>>& TF,
                               std::vector<std::pair<cv::Mat, cv::Mat>>& boardTF);

    //四元数转化
    static cv::Vec4f rotationMatrixToQuaternion(const cv::Mat& R);
    static void unifyQuaternionSigns(std::vector<cv::Vec4f>& quaternions);
    static cv::Mat quaternionToRotationMatrix(const cv::Vec4f& q);

public:
    //坐标变换中的默认条件，选择第一个板作为原点，其他板子的变化关系要转化成相对于主板的关系储存起来，查找时再进行计算

    TransformTool();
    ~TransformTool();

    //查找列表中从rest到main的变化关系
    static bool lookupTransform(const std::vector<std::pair<cv::Mat, cv::Mat>>& maskTF, int main, int rest,
                                       std::pair<cv::Mat, cv::Mat>& transform);
    //在已经建立了主板后，后期添加其他的板的处理函数 TODO：待测
    static bool tfADD(std::pair<cv::Mat, cv::Mat>& transform, std::vector<std::pair<cv::Mat, cv::Mat>>& maskTF,
                             int main, int rest);

    //对旋转矩阵转化成4元数，使用opencv平均后，再转换回来，平移矩阵直接算术平均 TODO：待测
    static  void tfAverage(const std::vector<std::pair<cv::Mat, cv::Mat>>& allTF, std::pair<cv::Mat, cv::Mat>& avTF);

    //对传入的单张图片的棋盘进行处理，得到每个棋盘对该相机的 R T
    static bool getTF(const Corners& corners, const std::vector<Chessboard>& boards,
                      const std::vector<cv::Mat>& chessboards, const CameraParams& cameraParam,
                      std::vector<std::pair<cv::Mat, cv::Mat>>& tf);
    //只比上面那个多了一个函数
    static bool getBoardTF(const Corners& corners, const std::vector<Chessboard>& boards,
                           const std::vector<cv::Mat>& chessboards, const CameraParams& cameraParam,
                           std::vector<std::pair<cv::Mat, cv::Mat>>& tf);
    //后面使用tfAverage进行处理，然后计算坐标变化关系；
};

