#include "fer.h"

int main()
{
	int feat_num = 531;
	int total_num = 25;

	cv::Mat M(total_num, feat_num, CV_32FC1);
	
	int i=1;
	for(; i<=total_num; ++i)
	{
		FER fer;
		
		std::string path = "./zhumingde/" + std::to_string((long long)i)+".jpg";
		cv::Mat img = cv::imread(path.c_str());
		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		// cv::equalizeHist(img, img);
		std::cout<< i << std::endl;
		
		// cv::imshow("raw",gray);
		// cv::waitKey();
		
		fer.face_detect_all(gray);
		
		// std::cout<< fer.FaceROI.width << " " << fer.FaceROI.height << std::endl;
		
		if(fer.FaceROI.width != fer.IMG_WIDTH && fer.FaceROI.height != fer.IMG_HEIGHT)
		{
			// std::cout<< "OK1" << std::endl;
			cv::Mat crop = fer.face_alignment(gray);
			std::cout<< "rows: " << crop.rows << " cols: " << crop.cols << std::endl;
			// cv::resize(crop, crop, cv::Size(150,150));
			cv::Mat feat = fer.feat_extract(crop);
			// std::cout<< "OK3" << std::endl;
			feat.copyTo(M.row(i-1));
			/*
			cv::cvtColor(crop, crop, cv::COLOR_GRAY2BGR);
		    for (unsigned int i = 0; i < 68; ++i)
			{
				int t1 = fer.faceshape.part(i).x();
				int t2 = fer.faceshape.part(i).y();
				cv::circle(crop, cv::Point(t1, t2), 2, cv::Scalar(0,0,255), -1);
			}
			cv::imshow("small", crop);
			cv::rectangle(gray,fer.FaceROI, cv::Scalar(0,0,255),1,1,0);
			cv::imshow("Raw", gray);
			cv::waitKey();
			*/
		}
		else
		{
			std::cout<< "NO."<< i << " image detecting failed!" << std::endl;
			break;
		}
	}
	if(i > total_num)
	{
		int labels[25] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};
		cv::Mat labelsMat(total_num, 1, CV_32SC1, labels);
		
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		// edit: the params struct got removed,
		// we use setter/getter now:
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR/*POLY*/);
		// svm->setGamma(3); 
   
		svm->train( M , cv::ml::ROW_SAMPLE , labelsMat );

		svm->save("zhumingde.xml");

		std::cout << "ok!" << std::endl;
	}
	
	return 0;
}
