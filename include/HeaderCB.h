/*  Copyright 2017 onlyliu(997737609@qq.com).                                */
/*                                                                        */
/*  part of source code come from https://github.com/qibao77/cornerDetect */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */
#pragma once

#include "opencv2/opencv.hpp"
#include <vector>

#define logd std::printf;

struct Corners {
	std::vector<cv::Point2f> p;
	std::vector<cv::Vec2f> v1;
	std::vector<cv::Vec2f> v2;
	std::vector<float> score;
};


struct CornerInfo {
	cv::Point2f p;
	int chessidx;
	int row, col;
	int idx;
	float x, y;
	bool vaild;
	int neardiskidx;
	float nearestdiskdistance;

	CornerInfo operator=(const CornerInfo& value) {
		p = value.p;
		chessidx = value.chessidx;
		row = value.row;
		col = value.col;
		idx = value.idx;
		x = value.x;
		y = value.y;
		vaild = value.vaild;
		neardiskidx = value.neardiskidx;
		nearestdiskdistance = value.nearestdiskdistance;

		return *this;
	}

	CornerInfo() {
		vaild = true;
		neardiskidx = -1;
		nearestdiskdistance = -1.0;
	}
};

struct ImageChessesStruct {
	std::vector<std::vector<cv::Point2f>> flagpostion;
	std::vector<std::vector<std::vector<int>>> idxconersflags;
	std::vector<int> choosecorneri;

	cv::Rect rt;
	int cbnum;
	std::vector<std::vector<CornerInfo>> chesscorners;
	bool flagbeginfromzero;

	ImageChessesStruct& operator=(ImageChessesStruct& value) {
		cbnum = value.cbnum;
		chesscorners = value.chesscorners;
		return *this;
	}
};

struct Chessboard {
	int id;
	std::pair<int, int> size;
	double squareLens;
	std::vector<cv::Point3f> objectPoints;

	Chessboard(const int id, const std::pair<int, int>& size, const double squareLens): id(id), size(size),
		squareLens(squareLens) {
		for (int i = 0; i < size.first; i++) {
			for (int j = 0; j < size.second; j++) {
				// 按实际棋盘格尺寸生成坐标，原点在棋盘格左上角
				objectPoints.emplace_back(j * squareLens, i * squareLens, 0.0f);
			}
		}
	}

	~Chessboard() = default;
};

struct CameraParams {
	int id;
	cv::Mat D;
	cv::Mat K;
	cv::Mat R;
	cv::Mat T;

	CameraParams() = default;
};