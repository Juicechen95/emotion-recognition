#include "fer.h"

int main()
{

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cerr << "Unable to connect to camera" << std::endl;
		return 1;
	}
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);

	FER fer;
	cv::Mat frame;
	// cv::Rect DrawRect;
	while(true){
	
		cap >> frame;
		
		cv::Mat img, gray, crop;
		img = frame.clone();
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		
		fer.face_detect(gray);
		
		if(fer.FaceROI.width != fer.IMG_WIDTH && fer.FaceROI.height != fer.IMG_HEIGHT)
		{
			crop = fer.face_alignment(gray);
			
		    cv::cvtColor(crop, crop, cv::COLOR_GRAY2BGR);
		    for (unsigned int i = 0; i < 68; ++i)
			{
				int t1 = fer.faceshape.part(i).x();
				int t2 = fer.faceshape.part(i).y();
				cv::circle(crop, cv::Point(t1, t2), 2, cv::Scalar(0,0,255), -1);
			}
			cv::imshow("small", crop);
		}
	
		cv::rectangle(img,fer.FaceROI, cv::Scalar(0,0,255),1,1,0);
		cv::imshow("Raw", img);
		
		if(cv::waitKey(30) == 27)	break;
	}

	return 0;
}
