#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include </home/yimeng/opencv-2.4.13/modules/features2d/include/opencv2/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <algorithm>

using namespace std;
using namespace cv;

class MulticascadeDetector{

public:
	String cascade_name;
	String video_name; 
	VideoCapture cap; 
	CascadeClassifier casClassifier;
	string window_name;
	RNG rng;
	std::vector<Rect> _lasttools;
	std::vector<Rect> _capturedtools;
	//Mat frame; 
	//Mat frame_gray;
	Mat cap_frame; 
	int loop; 
	int flag;

	// MATLAB re-creating
	vector< Rect > Bboxes; 
	vector<int> BoxIds; 
	vector<vector <int> > Points;
	vector<int> PointIds;
	vector<int> BoxScores; 
	vector<vector <int> > bbox1;
	int NextId; 
	int area; 
	int BoxIdx; 

	// testers
	Mat frame_gray;
    Mat Mask;
    Mat ROI;
    double MinHessian;
    int octaves;
    int octaveLayers;
    SurfFeatureDetector sDetector;
	Mat Key_frame; 
	vector<cv::KeyPoint> Keypoints;
	vector<cv::KeyPoint> nextKeypoints;

	vector <float> err;
    vector <uchar> Status;
    // Size WinSize;
    // int maxLevel;
    // TermCriteria termcrit;
    // int flags;
    // double minEigThreshold;
    Mat nextKey_frame; 
    vector< vector<int> > currentPoints; 
    vector< vector<int> > nextPoints; 
    vector<int> pIDs;
    vector<Point2f> currentPoints2f;
    vector<Point2f> nextPoints2f;
    vector<int> initer;
    Point2f point2fIniter; 
    KeyPoint keyPointIniter; 
    vector<int> sortedBboxesId; 


	MulticascadeDetector(){
		cascade_name = "toolDetector.xml";
		video_name = "testing_1.1.mov"; 
		loop = 0; 
		flag = 0; 
		Rect initRect(0,0,0,0); 
		for(int i = 0 ; i < 10; i++){
			Bboxes.push_back(initRect); 
		}
		
		std::vector<int> BoxIds(500); 
		vector<vector <int> > Points(500, vector<int>(4));
		std::vector<int> PointIds(500,8);
		std::vector<int> BoxScores(500); 
		NextId = 1;
		vector<vector <int> > bbox1(500, std::vector<int>(4)); 
		area = 0; 
		BoxIdx = 0; 

		vector <float> err(100);
	    vector <uchar> Status(100);
	    // Size WinSize = Size(31,31);
	    // int maxLevel = 3;
	    // TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.03);
	    // int flags = 0;
	    // double minEigThreshold = 1e-4;

	    for(int u = 0; u < 2; u++){
	    	initer.push_back(0); 
	    }
	    for(int u = 0; u < 1000; u++){
	    	currentPoints.push_back(initer); 	
	    }
	    // currentPoints = new vector< std::vector<int> >(1000, std::vector<int>(2));


	    // vector<int> initer(2, 0);
	    // for(int jj = 0; jj < 500; jj++){
	    // 	currentPoints.push_back(initer);
	    // }
    	for( int u = 0; u < 500; u++){
    		nextKeypoints.push_back(keyPointIniter);
    	}
    	
	}
	void detectProcess(){
 
		int i = 0; 
		Mat frame;
		Mat oldframe;
		window_name = "Capture";
		rng(12345);
		if( !casClassifier.load( cascade_name ) ){ printf("--(!)Error loading\n");};
		//casClassifier.load(cascade_name);//
		cout << "READ IN FRAMES STARTS" << endl; 
	        VideoCapture cap(video_name); 
	        if( !cap.isOpened() ){
		     printf("cap is not opened");
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, 10);
		cap >> frame; 
		cap >> cap_frame;

			while( 1 ){
			  //-- 3. Apply the classifier to the frame
			
			if( !frame.empty() ){ 
			    loop++; 
			    detectAndDisplay( frame ); 
			    sortedBboxesId = sortRectByXAndGiveBackIndexes(Bboxes); 
				for(int j = 0; j < (int)sortedBboxesId.size(); j++){
			    	cout <<"ORDER of the Bboxes " <<sortedBboxesId[j] << endl;
			    }
			    Bboxes = rearrangeBboxesUsingSortedIndexes(sortedBboxesId); 
			    
			}else{ 
			    
			    printf(" --(!) No captured frame -- Break! \n"); break; 
			}
				oldframe = frame; 
			    cap.read(frame); 
			    cap.read(cap_frame);

			SingleTracker(oldframe, frame);
			// member "nextPoints2f" is tracked points
			for( int j = 0; j < (int)Bboxes.size(); j++){
				cout << "the " << j << "th " << "Bboxes, " << ///
				"x= " << Bboxes[j].x << " y = " << Bboxes[j].y << endl;  
			}

			printf("%d \n", i); 
	    	i++;

			int c = waitKey(10);
			if( (char)c == 'c' ) { break; }
      		}
	}

	void SingleTracker(Mat _oldframe, Mat _frame){
		Mat oldframe_gray; 
		cvtColor( _oldframe, oldframe_gray, CV_BGR2GRAY );
	    equalizeHist( oldframe_gray, oldframe_gray );
	    Mat Mask = Mat::zeros(_oldframe.size(), CV_8U); 
	    Mat ROI(Mask, Bboxes[0]);// init the mask matrix
	    ROI = Scalar(255,255,255);
	    double MinHessian = 400;
	    int octaves = 3;
	    int octaveLayers = 6;
	    SurfFeatureDetector sDetector(MinHessian, octaves, octaveLayers);
		sDetector.detect(oldframe_gray, Keypoints, Mask);
		Mat Key_frame;
		drawKeypoints(_oldframe, Keypoints, Key_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("Features", Key_frame); 
		cout << "till opt start \n"; 

		// convert the Keypoints to Points
			KeyPoint::convert(Keypoints, currentPoints2f, pIDs);

		// cout<<"currentPoints : " << currentPoints2f[0].y <<endl;
		cout<<"1 : " << currentPoints2f[0].x << endl; 
		// 	cout <<"loop start";  
		for(int ii = 0; ii < (int)currentPoints2f.size(); ii++){
			cout<<"1 : " << ii << endl;
			currentPoints[ii][0] = (int)Keypoints[ii].pt.x;
			cout<<"2\n";
			currentPoints[ii][1] = (int)Keypoints[ii].pt.y;
			cout<<"3\n";
		}

		
		cout << "till opt start 2\n";
		Mat frame_gray;
		cvtColor( _frame, frame_gray, CV_BGR2GRAY );
	    equalizeHist( frame_gray, frame_gray );
	    Mat nextKey_frame;  
	    cout << "till opt start 3\n";
		/////drawKeypoints(frame_gray, nextKeypoints, nextKey_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	    //calcOpticalFlowPyrLK(oldframe_gray, frame_gray, currentPoints, nextPoints, Status, err, WinSize, maxLevel, termcrit, flags, minEigThreshold); 
	    calcOpticalFlowPyrLK(oldframe_gray, frame_gray, currentPoints2f, nextPoints2f, Status, err);//, WinSize, maxLevel, termcrit, flags, minEigThreshold); 
	    cout << "till opt end \n"; 
	    cout << "tracking points " << nextPoints2f.size() << endl;
	    for(int j = 0; j < (int)nextPoints2f.size(); j++){
			nextKeypoints[j].pt.x = nextPoints2f[j].x;
			nextKeypoints[j].pt.y = nextPoints2f[j].y;
		}
		cout << "tracking points " << nextKeypoints.size() << endl;
	    drawKeypoints(_frame, nextKeypoints, nextKey_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("nextFeatures", nextKey_frame);
	}
	/////
	// void addDetection(Mat currentFrame, vector<Rect> bboxes){ //seems tracker in CV is a function
	// 	//assume bboxes are already there
	// 	for(int j = 0; j < bboxes.size(), j++){
	// 		int boxIdx = findMatchingBox(bboxes[i]); 
	// 		if (boxIdx.empty()){
	// 			Bboxes.push_back(bboxes[j]);

	// 			// detector
	// 			vector<cv::KeyPoint> Keypoints;
	// 			double MinHessian = 400;
 //        	    int octaves = 3;
 //        	    int octaveLayers = 6;
 //        	    SurfFeatureDetector sDetector(MinHessian, octaves, octaveLayers);
 //        		Mat Mask = Mat::zeros(frame.size(), CV_8U); 
	// 		    Mat ROI(Mask, bboxes[j]);// init the mask matrix
	// 		    ROI = Scalar(255,255,255);
	// 		    sDetector.detect(currentFrame, Keypoints, Mask);
	// 		    cout << "size : " << Keypoints.size() << endl;

	// 		}
	// 		else{

	// 		}
	// 	}		
		
	// }


	int findMatchingBox(Rect box){
		for(int i = 0; i < (int)Bboxes.size(); i++){
			////
			area = computeRectJoinUnion(Bboxes[i], box);
			if(area > 0.2 * Bboxes[i].width * Bboxes[i].height){
				//BoxIdx = BoxIds[i]; // Here's the problem
				BoxIdx = 111; 
				return BoxIdx; 
			}
		}
		return 0; 
	}

private:
	
	

	/** @function detectAndDisplay */
	void detectAndDisplay( Mat _frame )
	{
	    std::vector<Rect> tools;

	    Mat frame_gray;

	    cvtColor( _frame, frame_gray, CV_BGR2GRAY );
	    equalizeHist( frame_gray, frame_gray );
	    int k = 0; 
	    //-- Detect faces
	    casClassifier.detectMultiScale( frame_gray, tools, 1.005, 2, 0|CASCADE_SCALE_IMAGE, Size(0, 0) );

	    for( size_t i = 0; i < tools.size(); i++ ){
	  	    
	  	    rectangle(_frame, Point(tools[i].x, tools[i].y), Point(tools[i].x + tools[i].width, tools[i].y + tools[i].height), Scalar(255,255,255)); 

	  	    Mat toolROI = frame_gray( tools[i] );
	  	    
	  	    printf("Tool detected in this frame. i = %d\n", (int)i + 1);
	  	    
	        if(loop%10 == 0 && computeRectJoinUnion(tools[i], _lasttools[i]) > 0.9 ){
			k++;
			_capturedtools = tools; 
	  	        //rectangle(cap_frame, Point(tools[i].x, tools[i].y), Point(tools[i].x + tools[i].width, tools[i].y + tools[i].height), Scalar(0,0,255)); 
			printf("Tool TRACKING TRACKING in this frame. k = %d, tool number is %d\n", (int)k, (int)i + 1);
			flag = 1; 	  	    
			} 
		if(flag  == 1){
			rectangle(cap_frame, Point(_capturedtools[i].x, _capturedtools[i].y), Point(_capturedtools[i].x + _capturedtools[i].width, _capturedtools[i].y + _capturedtools[i].height), Scalar(0,0,255));
		}
		k = 0;
	        _lasttools = tools; 

	        // test codes
	        Bboxes = tools; 
	        //int testerID = 0;
	        //testerID = findMatchingBox(tools[1]); 
	        //cout << "testerID: " << testerID << "\n"; 
	    	
	    	
	    }
	    //-- Show what you got
	    imshow( window_name, _frame );
	    imshow( "captured", cap_frame); 
	 }
	 // bool compareRect(const Rect &A, const Rect &B){
	 // 	return A.y < B.y;
	 // }

	 // rearrange Bboxes to have a rising order
	//qsort(Bboxes, Bboxes.size(), sizeof(Rect), compareRect); 
	//std::sort(Bboxes.begin(), Bboxes.end(), compareRect);
	vector<int> sortRectByXAndGiveBackIndexes(vector<Rect> _Bboxes){
		vector<int> _sortedId; 
		for(int u = 0; u < 10; u++){
			_sortedId.push_back(u+1);
		}
		std::vector<int> _B;
		for(int u = 0; u < (int)_Bboxes.size(); u++){
			_B.push_back(_Bboxes[u].x); 
		}
		cv::sortIdx(_B, _sortedId, CV_SORT_ASCENDING);
		return _sortedId;
	}
	// cv::sortIdx(Bboxes, sortedBboxesId, CV_SORT_ASCENDING);
	// for(int j = 0; j < (int)sortedBboxesId.size(); j++){
	// 	cout << "Bboxes in ascending order No. "  <<sortedBboxesId[j] << endl;
	// }

	vector<Rect> rearrangeBboxesUsingSortedIndexes(vector<int> Indexes){
		vector<Rect> newBboxes; 
		Rect initRect(0,0,0,0); 
		for(int u = 0; u < (int)Indexes.size(); u++){
			newBboxes.push_back(initRect); 
		}
		for(int u = 0; u < (int)Indexes.size(); u++){
			newBboxes[u] = Bboxes[Indexes[u]];
		}
		return newBboxes; 
	}
	// this function computes the ratio of two consecutive bounding boxes
	float computeRectJoinUnion(const Rect &rc1, const Rect &rc2)
	{
	    CvPoint p1, p2;                 
	    p1.x = std::max(rc1.x, rc2.x);
	    p1.y = std::max(rc1.y, rc2.y);

	    p2.x = std::min(rc1.x +rc1.width, rc2.x +rc2.width);
	    p2.y = std::min(rc1.y +rc1.height, rc2.y +rc2.height);
	    // here p1 and p2 are computed to be the diagnal points of the overlapping rectangulars

	    float AJoin = 0;
	    if( p2.x > p1.x && p2.y > p1.y ) // if overlapse            
	    {
	        AJoin = (p2.x - p1.x)*(p2.y - p1.y);    
	    }
	    float A1 = rc1.width * rc1.height;
	    float A2 = rc2.width * rc2.height;
	    float AUnion = (A1 + A2 - AJoin);                 
	    //float edge_ratio = ((float)rc1.width/((float)rc1.width + (float)rc2.width) ) * ( (float)rc1.height/((float)rc1.height + (float)rc2.height));
	    if( AUnion > 0 ){
		printf("In the tracking mode, ratio = %f\n", (float)(AJoin * AJoin)/(A1*A2));
	        return AJoin;
		
		}                   
	    else{
		printf("Nothing can be tracked in this frame! \n");
		//printf("abs_ratio = %f \n", edge_ratio); 
	        return 0;
		}
	}

};
