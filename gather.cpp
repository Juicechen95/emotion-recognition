// #include <sys/stat.h> 　
// #include <sys/types.h> 
#include "fer.h"
#include<iostream>

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
	int count = 0;
	int sample_num = 5; // number of training samples for each emotion of a person.
	std::cout <<"请输入姓名： " <<std::endl;
	std::cin>>fer.usrname;
	std::cout <<"请依次摆出如下表情：无表情，开心，愤怒，悲伤，惊讶。并各自截取5张图片，按任意字母可截取图片。 " <<std::endl;
	while(true){
		
		cap >> frame;
		
		cv::Mat img, gray, gray1, gray2;
		img = frame.clone();
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		
		// cv::medianBlur(gray, gray1, 7);
		cv::equalizeHist(gray, gray2);
		
		fer.face_detect(gray2);
		bool hasFace = false;
		if(fer.FaceROI.width != fer.IMG_WIDTH && fer.FaceROI.height != fer.IMG_HEIGHT)
		{
			cv::rectangle(gray, fer.FaceROI, cv::Scalar(0,0,255),1,1,0);
			hasFace = true;
		}
		
		
		// cv::imshow("gray1", gray1);
		// cv::imshow("gray2", gray2);
		//std::string show_res = "neturel: 1" ;
		//cv::putText(gray, show_res.c_str(), cv::Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
		cv::imshow("gray", gray);
		if(cv::waitKey(20) != 255 && hasFace)	
		{
			count++;
			//if(count < (sample_num*1+1))
			//{
				//std::string show_res = "happy: " ;
				//cv::putText(gray, show_res.c_str(), cv::Point(20, 40), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			//}
			std::string path = "./zhumingde/" + std::to_string((long long)count)+".jpg";
			cv::imwrite(path, gray2);
			std::cout << count << std::endl;
		}
	}
	
	return 0;
}
