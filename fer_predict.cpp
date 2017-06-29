#include "fer.h"
#include <deque>

void Find_max_label(std::deque<int> &res, int expression[])
{
	for(std::deque<int>::iterator it = res.begin(); it != res.end(); ++it)
	{
		expression[*it]++;
	}
}

int main()
{
	const int catche = 10;
	std::deque<int> res(catche, 1);
	float possitive = 0;
	float negtive = 0;
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

	
	 //cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("general.xml");
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("ray_new.xml");
	cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("zhumingde.xml");
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("jaffe.xml");
	
	while(true){
	
		cap >> frame;
		
		cv::Mat img, gray, crop;
		img = frame.clone();
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		fer.face_detect(gray);
		
		if(fer.FaceROI.width != fer.IMG_WIDTH && fer.FaceROI.height != fer.IMG_HEIGHT)
		{
			crop = fer.face_alignment(gray);
			// std::cout<< "rows: " << crop.rows << " cols: " << crop.cols << std::endl;
		    cv::Mat feat = fer.feat_extract(crop);
		    // std::cout << "feat.cols: " << feat.cols << " " << svm->getVarCount() << std::endl;
			//cv::Mat res;   // 输出
			// svm->predict(feat, res);
			
			res.pop_front();
			res.push_back((int)svm->predict(feat));
			int expression[10] = {0};
			Find_max_label(res, expression);
			float neturel = ((float)expression[1]/catche), smile = ((float)expression[2]/catche), anger = ((float)expression[3]/catche), sad = ((float)expression[4]/catche), surprise = ((float)expression[5]/catche);
			if (anger ==0 && sad == 0)
			{
				possitive = 2-neturel-surprise;
			}
			else
			{
				possitive = smile/(anger+sad)-neturel-surprise;
			}
			if (smile == 0)
			{
				negtive = 2-neturel-surprise;
			}
			else
			{
				negtive = (anger+sad)/smile-neturel-surprise;
			}
			// std::cout << svm->predict(feat) << std::endl;
			//double font_scale = 2; 
			std::string show_res = "neutral: " ;//+ std::to_string(neturel);
			cv::putText(img, show_res, cv::Point(30, 20), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			show_res = "happy: " ;//+ std::to_string(smile);
			cv::putText(img, show_res.c_str(), cv::Point(45, 40), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			show_res = "anger: " ;//+ std::to_string(anger);
			cv::putText(img, show_res.c_str(), cv::Point(50, 60), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			show_res = "sad: " ;//+ std::to_string(sad);
			cv::putText(img, show_res.c_str(), cv::Point(78, 80), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			show_res = "surprise: " ;//+ std::to_string(surprise);
			cv::putText(img, show_res.c_str(), cv::Point(20, 100), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255,0,0));
			
			cv::rectangle(img ,cv::Point(140, 10), cv::Point(140+neturel*100, 15), cv::Scalar(0,255,0),6,1,0);
			cv::rectangle(img ,cv::Point(140, 30), cv::Point(140+smile*100, 35), cv::Scalar(0,255,0),6,1,0);
			cv::rectangle(img ,cv::Point(140, 50), cv::Point(140+anger*100, 55), cv::Scalar(0,255,0),6,1,0);
			cv::rectangle(img ,cv::Point(140, 70), cv::Point(140+sad*100, 75), cv::Scalar(0,255,0),6,1,0);
			cv::rectangle(img ,cv::Point(140, 90), cv::Point(140+surprise*100, 95), cv::Scalar(0,255,0),6,1,0);
			if (possitive>1)
			{
				show_res = "Positive emotion" ;
				cv::putText(img, show_res.c_str(), cv::Point(450, 40), CV_FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0,0,255));
			}
			else if(negtive >1)
			{
				show_res = "Negative emotion" ;
				cv::putText(img, show_res.c_str(), cv::Point(450, 40), CV_FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0,0,255));
			}
			else
			{
				show_res = "Neutral emotion" ;
				cv::putText(img, show_res.c_str(), cv::Point(450, 40), CV_FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0,0,255));
			}
			cv::cvtColor(crop, crop, cv::COLOR_GRAY2BGR);
		    for (unsigned int i = 0; i < 68; ++i)
			{
				int t1 = fer.faceshape.part(i).x();
				int t2 = fer.faceshape.part(i).y();
				cv::circle(crop, cv::Point(t1, t2), 2, cv::Scalar(0,0,255), -1);
			}
			
			// 鼻中
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(27).x()-5, fer.faceshape.part(27).y(), 10, 30), cv::Scalar(0,255,0),1,1,0);
			// 鼻左
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(31).x()-5, fer.faceshape.part(31).y()-40, 10, 30), cv::Scalar(255,0,0),1,1,0);
			// 鼻右
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(35).x()-5, fer.faceshape.part(35).y()-40, 10, 30), cv::Scalar(0,0,255),1,1,0);
			// 嘴中
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(33).x()-20, fer.faceshape.part(33).y()+10, 40, 40), cv::Scalar(0,255,0),1,1,0);
			// 嘴左
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(48).x()-15, fer.faceshape.part(48).y()-40, 20, 40), cv::Scalar(255,0,0),1,1,0);
			// 嘴右
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(54).x()-5, fer.faceshape.part(54).y()-40, 20, 40), cv::Scalar(0,0,255),1,1,0);
			// 左眼
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(36).x()-10, fer.faceshape.part(36).y()-10, 20, 20), cv::Scalar(255,0,0),1,1,0);
			// 左眼2
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(39).x()-5, fer.faceshape.part(39).y()-5, 10, 10), cv::Scalar(255,0,0),1,1,0);
			// 右眼
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(45).x()-10, fer.faceshape.part(45).y()-10, 20, 20), cv::Scalar(0,0,255),1,1,0);
			// 右眼2
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(42).x()-5, fer.faceshape.part(42).y()-5, 10, 10), cv::Scalar(0,0,255),1,1,0);
			// 左眉
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(36).x(), fer.faceshape.part(36).y()-30, 40, 10), cv::Scalar(255,0,0),1,1,0);
			// 右眉
			cv::rectangle(crop,cv::Rect(fer.faceshape.part(45).x()-40, fer.faceshape.part(45).y()-30, 40, 10), cv::Scalar(0,0,255),1,1,0);
			
			
			cv::imshow("small", crop);
		}
		cv::rectangle(img,fer.FaceROI, cv::Scalar(0,0,255),1,1,0);
		cv::imshow("Raw", img);
		if(cv::waitKey(30) == 27)	break;
	}

	return 0;
}
