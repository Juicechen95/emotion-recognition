#pragma once

#include <vector>
#include <algorithm>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/ml/ml.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include <direct.h>

// using namespace dlib;
// using namespace std;

class FER
{
private:
	dlib::frontal_face_detector FaceDetector;
	dlib::shape_predictor pose_model;
	
	unsigned short NotFoundCount;
	unsigned int FaceDetectNormalizedW;
	unsigned int FaceDetectCommonFscale;
	
	cv::Rect SearchROI;
	
	std::vector<float> get_features(cv::Mat submat, int row_ratio, int col_ratio);
	
public:

	unsigned int IMG_WIDTH;
	unsigned int IMG_HEIGHT;
	char usrname;
	cv::Rect FaceROI;
	dlib::full_object_detection faceshape;

	FER();
	~FER();
	cv::Rect face_detect(cv::Mat SearchArea);
	cv::Rect face_detect_all(cv::Mat SearchArea);
	cv::Mat face_alignment(cv::Mat gray);
	
	// cv::Mat face_rotate();
	cv::Mat feat_extract(cv::Mat &face);
	// fer_train();
	// fer_predict();
};
