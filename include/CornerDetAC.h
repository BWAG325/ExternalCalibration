/*  Copyright 2017 onlyliu(997737609@qq.com).                                */
/*                                                                        */
/*  part of source code come from https://github.com/qibao77/cornerDetect */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the paper: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */


#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "HeaderCB.h"

#if 1
#define  mtype CV_32F
#define  dtype  float
#else
#define  mtype CV_64F
#define dtype  double
#endif

class CornerDetAC
{
public:
	CornerDetAC();
	~CornerDetAC();
	void detectCorners(cv::Mat &Src, Corners& mCorners, dtype scoreThreshold, bool isRefine = true);
	void chessboardsFromCorners(std::vector<std::vector<cv::Point2f>>chessboards);
private:
	
	static dtype normalPDF(dtype dist, dtype mu, dtype sigma);

	static void getMin(cv::Mat src1, cv::Mat src2, cv::Mat &dst);
	
	static void getMax(cv::Mat src1, cv::Mat src2, cv::Mat &dst);
	
	static void getImageAngleAndWeight(const cv::Mat& img, cv::Mat &imgDu, cv::Mat &imgDv, cv::Mat &imgAngle, cv::Mat &imgWeight);
	
	void edgeOrientations(cv::Mat imgAngle, cv::Mat imgWeight, int index);
	
	static void findModesMeanShift(std::vector<dtype> hist, std::vector<dtype> &hist_smoothed, std::vector<std::pair<dtype, int>> &modes, dtype sigma);
	
	void scoreCorners(const cv::Mat& img, const cv::Mat& imgAngle, const cv::Mat& imgWeight, std::vector<cv::Point2f> &corners, std::vector<int> radius, std::vector<float> &score);
	
	static void cornerCorrelationScore(const cv::Mat& img, cv::Mat imgWeight, std::vector<cv::Point2f> cornersEdge, float &score);
	
	void refineCorners(std::vector<cv::Point2f> &corners, cv::Mat imgDu, cv::Mat imgDv, const cv::Mat& imgAngle, const cv::Mat& imgWeight, float radius);
	
	static void createKernel(float angle1, float angle2, int kernelSize, cv::Mat &kernelA, cv::Mat &kernelB, cv::Mat &kernelC, cv::Mat &kernelD);
	
	void nonMaximumSuppression(cv::Mat& inputCorners, std::vector<cv::Point2f>& outputCorners, int patchSize, dtype threshold, int margin);
	static float norm2d(cv::Point2f o);
	std::vector<cv::Point2f> templateProps;
	std::vector<int> radius;
	std::vector<cv::Point2f> cornerPoints;
	std::vector<std::vector<dtype>> cornersEdge1;
	std::vector<std::vector<dtype> > cornersEdge2;
	std::vector<cv::Point2f* > cornerPointsRefined;
	
};

