#include "fer.h"
#include <fstream>
int main()
{
	int feat_num = 531;
	int total_num = 30;
	int exp_num = 5;

	cv::Mat M(total_num*exp_num, feat_num, CV_32FC1);
	int labels[150] = {0};
	
	bool correct = true;
	
	for(int j=1; j<=exp_num; ++j)
	{
		std::string d_path = "./jaffe/" + std::to_string((long long)j) + "/";
		std::ifstream fin(d_path + "/info.txt");
		std::cout << "d_path: " << d_path + "info.txt" << std::endl;
		int i=1;
		for(; i<=total_num; ++i)
		{
			FER fer;
			std::string filename;
			std::getline(fin, filename);
			std::string path = d_path + filename;
			std::cout << "filename: " << filename << std::endl;
			cv::Mat img = cv::imread(path.c_str());
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			cv::resize(gray, gray, cv::Size(640, 480));
			cv::equalizeHist(gray, gray);
			
			std::cout<< "Expression: " << j << " NO. " << i << std::endl;
		
		
			fer.face_detect_all(gray);
		
			if(fer.FaceROI.width != fer.IMG_WIDTH && fer.FaceROI.height != fer.IMG_HEIGHT)
			{
				cv::Mat crop = fer.face_alignment(gray);
				// std::cout<< "rows: " << crop.rows << " cols: " << crop.cols << std::endl;
				cv::Mat feat = fer.feat_extract(crop);
				feat.copyTo(M.row((j-1)*total_num+i-1));
				labels[(j-1)*total_num+i-1] = j;
			}
			else
			{
				std::cout<< "Expression: " << j << "NO."<< i << " image detecting failed!" << std::endl;
				break;
			}
		}
		fin.close();
		if(i <= total_num)
		{
			correct = false;
			break;
		}
	}
	if(correct)
	{
		// int labels[25] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};
		cv::Mat labelsMat(total_num*exp_num, 1, CV_32SC1, labels);
		
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		// edit: the params struct got removed,
		// we use setter/getter now:
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR/*POLY*/);
		// svm->setGamma(3); 
   
		svm->train( M , cv::ml::ROW_SAMPLE , labelsMat );

		svm->save("jaffe.xml");

		std::cout << "ok!" << std::endl;
	}
	
	return 0;
}
