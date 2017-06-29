#include "fer.h"

FER::FER()
{
	IMG_HEIGHT = 480;
	IMG_WIDTH = 640;
	
	NotFoundCount = 0;
	FaceDetectNormalizedW = 110;
	FaceDetectCommonFscale = 2;
	
	SearchROI = cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
	FaceROI = cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
	
	// 载入人脸检测和特征点定位模型
	FaceDetector = dlib::get_frontal_face_detector();
	
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

}

cv::Rect FER::face_detect_all(cv::Mat tmp)
{
	
	// int w = tmp.cols, h = tmp.rows;
	// double fscale = 1;
	// cv::resize(tmp, tmp, cv::Size(w / fscale, h / fscale));
	// cv::imshow("", tmp);
	// cv::waitKey();
	dlib::cv_image<unsigned char> cimg(tmp);
	std::vector<dlib::rectangle> faces = FaceDetector(cimg);
	std::cout << "faces.size: " << faces.size() << std::endl;
	if (faces.size() > 0)
	{
		auto face = faces[0];
		/*
		face.set_bottom(fscale * face.bottom());
		face.set_left(fscale *face.left());
		face.set_right(fscale *face.right());
		face.set_top(fscale *face.top());
		*/
		FaceROI = cv::Rect(face.left(), face.top(), face.width(), face.height()) & cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
	}
	
	return FaceROI;
}

cv::Rect FER::face_detect(cv::Mat SearchArea)
{
	// std::cout<<"ok1"<<std::endl;
	cv::Mat tmp = SearchArea(SearchROI);
	// std::cout<<"ok2"<<std::endl;
	int w = tmp.cols, h = tmp.rows;
	double fscale = w / static_cast<double>(FaceDetectNormalizedW);
	
	if (NotFoundCount != 0)	fscale = FaceDetectCommonFscale;
	cv::resize(tmp, tmp, cv::Size(w / fscale, h / fscale));
	
	dlib::cv_image<unsigned char> cimg(tmp);
	std::vector<dlib::rectangle> faces = FaceDetector(cimg);
	
	if (faces.size() > 0)
	{
		auto face = faces[0];
		face.set_bottom(fscale * face.bottom());
		face.set_left(fscale *face.left());
		face.set_right(fscale *face.right());
		face.set_top(fscale *face.top());
		
		FaceROI = cv::Rect(SearchROI.x + face.left(), SearchROI.y + face.top(), face.width(), face.height()) & cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
		
		int spw = face.width()*0.25;
		int sph = face.height()*0.25;

		SearchROI = cv::Rect(
			FaceROI.x - spw,
			FaceROI.y - sph,
			face.width() + 2 * spw, face.height() + 2 * sph) & cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
		// ToExtractFratures = true;
		NotFoundCount = 0;
		return FaceROI;
	}
	else
	{
		
		// Shapes.clear();
		// DrawRect = SearchROI;
		
		///扩展搜索区
		SearchROI = cv::Rect(IMG_WIDTH*(0.25 - 0.1*NotFoundCount), IMG_HEIGHT*(0.25 - 0.1*NotFoundCount), (0.5 + 0.2*NotFoundCount)*IMG_WIDTH, (0.5 + 0.2*NotFoundCount)*IMG_HEIGHT)& cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
		 if (NotFoundCount < 3)	++NotFoundCount;
		 FaceROI = cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
	}

	return FaceROI;
}

FER::~FER()
{
	
}

// 调用前保证一定有人脸
cv::Mat FER::face_alignment(cv::Mat gray)
{
	///提取脸部特征点
	dlib::cv_image<unsigned char> cimg(gray);
	std::vector<dlib::full_object_detection> tmpshapes;
	tmpshapes.push_back(pose_model(cimg, dlib::rectangle(FaceROI.x, FaceROI.y, FaceROI.x + FaceROI.width, FaceROI.y + FaceROI.height)));
	
	faceshape = tmpshapes[0];
	
	///人脸校正
	dlib::matrix<unsigned char> RotatedImg;
	auto chip = get_face_chip_details(faceshape);
	dlib::extract_image_chip(cimg, chip, RotatedImg);
	cv::Mat tmpRotatedFace(RotatedImg.nr(), RotatedImg.nc(),CV_8UC1, RotatedImg.begin());

	cv::Mat FaceArea = tmpRotatedFace.clone();
	faceshape = dlib::map_det_to_chip(faceshape, chip);
	
	return FaceArea;
}

std::vector<float> FER::get_features(cv::Mat submat, int row_ratio, int col_ratio)
{
	int w = submat.cols;
	int h = submat.rows;
	cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(w, h), cv::Size(w, h), cv::Size(w, h), cv::Size(w/col_ratio, h/row_ratio), 9);
	
	std::vector<float> descriptors;//结果数组

	hog->compute(submat, descriptors, cv::Size(1, 1), cv::Size(0, 0));
	return descriptors;
}


cv::Mat FER::feat_extract(cv::Mat &face)
{
	std::vector<float>  v1;
	std::vector<float> tmp;
	// std::cout<<"22y"<<faceshape.part(22).y()<<" 22x"<<faceshape.part(22).x()<<std::endl;
	// 鼻中
	tmp = get_features(face(cv::Rect(faceshape.part(27).x()-5, faceshape.part(27).y(), 10, 30)), 3, 1);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 鼻左
	tmp = get_features(face(cv::Rect(faceshape.part(31).x()-5, faceshape.part(31).y()-40, 10, 30)), 3, 1);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 鼻右
	tmp = get_features(face(cv::Rect(faceshape.part(35).x()-5, faceshape.part(35).y()-40, 10, 30)), 3, 1);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 嘴中
	// tmp = get_features(face(cv::Rect(faceshape.part(34).x()-20, faceshape.part(34).y()+15, 40, std::min(40, (int)(face.rows-faceshape.part(34).y()-15)), 1, 1);
	tmp = get_features(face(cv::Rect(faceshape.part(33).x()-20, faceshape.part(33).y()+10, 40, 40)), 4, 4);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 嘴左
	tmp = get_features(face(cv::Rect(faceshape.part(48).x()-15, faceshape.part(48).y()-40, 20, 40)), 4, 2);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 嘴右
	tmp = get_features(face(cv::Rect(faceshape.part(54).x()-5, faceshape.part(54).y()-40, 20, 40)), 4, 2);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 左眼
	tmp = get_features(face(cv::Rect(faceshape.part(36).x()-10, faceshape.part(36).y()-10, 20, 20)), 2, 2);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 左眼2
	tmp = get_features(face(cv::Rect(faceshape.part(39).x()-5, faceshape.part(39).y()-5, 10, 10)), 1, 1);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 右眼
	tmp = get_features(face(cv::Rect(faceshape.part(45).x()-10, faceshape.part(45).y()-10, 20, 20)), 2, 2);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 右眼2
	tmp = get_features(face(cv::Rect(faceshape.part(42).x()-5, faceshape.part(42).y()-5, 10, 10)), 1, 1);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 左眉
	tmp = get_features(face(cv::Rect(faceshape.part(36).x(), faceshape.part(36).y()-30, 40, 10)), 1, 4);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	// 右眉
	tmp = get_features(face(cv::Rect(faceshape.part(45).x()-40, faceshape.part(45).y()-30, 40, 10)), 1, 4);
	v1.insert(v1.end(), tmp.begin(), tmp.end());
	tmp.clear();
	
	cv::Mat Mfinal = cv::Mat(1, 531, CV_32FC1);
	memcpy(Mfinal.data, v1.data(), v1.size()*sizeof(float));
	
	return Mfinal;
}



















