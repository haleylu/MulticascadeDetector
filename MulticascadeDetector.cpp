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
//test git
//test git 2
using namespace std;
using namespace cv;
#define NOTHING -1
#define AREA_RATIO 0.7
#define SIZE_OF_ALL_POINTS 8000
#define SIZE_OF_ALL_BBOXES 800

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

	Mat cap_frame; 
	int loop; 
	int flags;

	int minX;
	int minY;
	int maxX;
	int maxY; 
	// MATLAB re-creating
	 
	vector<Rect> bboxes; 

	vector< Rect > Bboxes;
	vector<int> BoxIds; 
	vector<int> PointIds;
	vector<int> PointNums; 
	vector<int> PointNumsTillThisBbox; 
	vector<int> BoxScores; 

	vector<vector <int> > bbox1;
	int nextId; 
	int area; 
	int BoxIdx; 

	
	
    SurfFeatureDetector sDetector;
	vector<cv::KeyPoint> Keypoints;
	vector<cv::KeyPoint> nextKeypoints;
	vector<cv::KeyPoint> allNextKeypoints; 

	vector <float> err;
    vector <uchar> Status;

    vector< vector<int> > nextPoints; 
    vector<int> pIDs;
    vector<Point2f> currentPoints2f;
    vector<Point2f> nextPoints2f;
    vector<int> initer;
    Point2f point2fIniter; 
    KeyPoint keyPointIniter; 
    vector<int> sortedBboxesId; 

    int RedetectPointsFlag; 
    cv::KeyPoint nextOneKeyPoint; 
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    ////////////     initializer                                           ///////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
	MulticascadeDetector(){

		cascade_name = "toolDetector.xml";
		video_name = "testing_1.1.mov"; 
		window_name = "Capture";
		rng(12345);
		loop = 0; 
		flags = 5;
		RedetectPointsFlag = 0; 

		initTracker();
    	initDetector();
	}


	void initTracker(){
		Rect initRect(0,0,0,0); 
		// for(int i = 0 ; i < 5; i++){
		// 	Bboxes.push_back(initRect); 
		// 	bboxes.push_back(initRect);
		// }
		
		nextId = 1;
		vector<vector <int> > bbox1(500, std::vector<int>(4)); 
		area = 0; 
		BoxIdx = 0; 

	    for(int u = 0; u < 2; u++){
	    	initer.push_back(0); 
	    }

    	nextOneKeyPoint.pt.x = 0;
    	nextOneKeyPoint.pt.y = 0; 

    	for( int u = 0; u < SIZE_OF_ALL_POINTS; u++){
    		allNextKeypoints.push_back(keyPointIniter);
    	}
    	allNextKeypoints.clear();
    	for( int u = 0; u < SIZE_OF_ALL_BBOXES; u++){
    		Bboxes.push_back(initRect); 
    		BoxIds.push_back(0); 
    		PointIds.push_back(0); 
    		PointNums.push_back(0);
    		PointNumsTillThisBbox.push_back(0);
    		BoxScores.push_back(0);
    	}
    	Bboxes.clear(); 
    	BoxIds.clear();
    	PointIds.clear();
    	PointNums.clear();
    	PointNumsTillThisBbox.clear();
    	BoxScores.clear();
	}

	void initDetector(){
		vector <float> err(100);
	    vector <uchar> Status(100);
	}


	void detectProcess(){
 		
 		//settings for detect
		int ii = 0; 
		Mat frame;
		Mat oldframe;
		

		if( !casClassifier.load( cascade_name ) ){ printf("--(!)Error loading\n");};

		cout << "READ IN FRAMES STARTS" << endl; 
	        VideoCapture cap(video_name); 
	        if( !cap.isOpened() ){
		     printf("cap is not opened");
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, 10);

		// avoid core dump
		cap >> frame; 
		cap_frame = frame.clone();

			while( 1 ){
			cout << endl << endl; 
			printf("The %dth frame started. \n", ii); 
	    	ii++;
	    	oldframe = frame.clone(); 
			cap >> frame; 
	    	cap_frame = frame.clone();
			
	    	cout << "All togeether " << bboxes.size() << " bboxes" << endl; 
		    // Find(only once) and track the feature points in two consecutive frames
			// member "nextPoints2f" is tracked points

	    	//if(i == 30){break;} //debug you
	    	if(RedetectPointsFlag == 0){
	    		RedetectPointsFlag = 1; // close the Redetect status
				if( !frame.empty() ){ 
				    loop++; 
				    // Detect bounding boxes in the frame, save the corrdinates in Bboxes
				    // then display the Bboxes
				    detectAndDisplay( frame ); 
				    // Rearrange Bboxes by left top point.x
				    // sortedBboxesId = sortRectByXAndGiveBackIndexes(bboxes); 
				    
				    // bboxes = rearrangeBboxesUsingSortedIndexes(sortedBboxesId); 
				    
				    
				}else{ 
				    printf(" --(!) No captured frame -- Break! \n"); break; 
				}
				

				for(int j = 0; j < (int)bboxes.size(); j++){
					addDetection(oldframe, frame, bboxes[j]); 
					cout << "After addDetection(), Bboxes has size = " << Bboxes.size() << endl;
					cout << "--------------------" << endl;
				}
			}
			for(int j = 0; j < (int)Bboxes.size(); j++){
				SingleTracker(oldframe, frame, j);
			}
		 	
			// draw all the point at a time
			Mat nextKey_frame; 
			drawKeypoints(frame, allNextKeypoints, nextKey_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			imshow("nextFeatures", nextKey_frame);
			nextKey_frame.release();

			for( int j = 0; j < (int)Bboxes.size(); j++){
				cout << "the " << j << "th " << "Bboxes, " << ///
				"x= " << Bboxes[j].x << " y = " << Bboxes[j].y << " width= " << Bboxes[j].width << " height= " << Bboxes[j].height << endl;  
			}
			cout << "allNextKeypoints.size() = " << allNextKeypoints.size() << endl; 

			
			if(ii % 10 == 0){
				RedetectPointsFlag = 0; 
			}

			int c = waitKey(10);
			if( (char)c == 'c' ) { break; }

			getchar();
      		} 


	}

	void SingleTracker(Mat _oldframe, Mat _frame, int BboxNum){
		
		if(RedetectPointsFlag == 0){
	    	// move this part to addDetection() 
	    	

	    }else{
	     	// prepare oldframe_gray
			Mat oldframe_gray; 
			cvtColor( _oldframe, oldframe_gray, CV_BGR2GRAY );
		    equalizeHist( oldframe_gray, oldframe_gray );
		    // prepare frame_gray
		    Mat frame_gray;
			cvtColor( _frame, frame_gray, CV_BGR2GRAY );
		    equalizeHist( frame_gray, frame_gray );
	    	Keypoints.clear();
	    	currentPoints2f.clear(); 
		    nextPoints2f.clear();
		    nextKeypoints.clear();


		    cout << "(int)PointNumsTillThisBbox.size()  " << (int)PointNumsTillThisBbox.size() << endl;  
		    cout << "PointNumsTillThisBbox[BboxNum] = " <<PointNumsTillThisBbox[BboxNum] <<" BboxNum = " <<BboxNum<< endl; 
		    cout << "PointNumsTillThisBbox[BboxNum + 1 ] = " <<PointNumsTillThisBbox[BboxNum + 1]<< endl; 
		    	// use allNextPoints to reconstruct the points for tracker
		    	for(int j = 0; j < PointNumsTillThisBbox[BboxNum + 1] - PointNumsTillThisBbox[BboxNum]; j++){
		    		//cout << "BboxNum = " << BboxNum << endl; 
		    		Keypoints.push_back(allNextKeypoints[PointNumsTillThisBbox[BboxNum]  + j ]);

		    	}
		    	// yue jie zai ci
		    	cout << "Keypoints Number is " << Keypoints.size() << endl;
			
	  //   	Mat debug_frame; 
	  //   	debug_frame = _frame.clone(); 
	  //    	drawKeypoints(_frame, Keypoints, debug_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			// rectangle(debug_frame, Point(Bboxes[BboxNum].x, Bboxes[BboxNum].y), Point(Bboxes[BboxNum].x + Bboxes[BboxNum].width, Bboxes[BboxNum].y + Bboxes[BboxNum].height), Scalar(255,255,255)); 
			// imshow("debugFeatures before", debug_frame);
			// getchar();

		   
			// convert the Keypoints to Points2f for optical flow tracking
			KeyPoint::convert(Keypoints, currentPoints2f, pIDs);
			// optical flow, track currentPoints2f, give out nextPoints2f	    
			calcOpticalFlowPyrLK(oldframe_gray, frame_gray, currentPoints2f, nextPoints2f, Status, err);
		    
		    // for(int e = 0; e < (int)currentPoints2f.size(); e ++){
		    // 	cout << "OLD.x " << currentPoints2f[e].x << ", OLD.y " << currentPoints2f[e].y << endl; 
		    // 	cout << "NEW.x " << nextPoints2f[e].x << ", NEW.y " << nextPoints2f[e].y << endl; 
		    // 	cout << "err " << err[e] << endl;
		    // 	if(err[e] > 20){

		    // 		nextPoints2f[e].x = currentPoints2f[e].x;
		    // 		nextPoints2f[e].y = currentPoints2f[e].y;
		    // 		err[e] = 20; 
		    // 	}
		    // }
		    int boxIdx = BoxIds[BboxNum]; 
		    
		    // Rect _deletedBbox = Bboxes[BboxNum]; 
		     
		    minX = 10000; 
		    maxX = -10000;
		    minY = 10000;
		    maxY = -10000;
		    // update nextKeyPoints, allNextKeypoints, PointIds using nextPoints2f
		    for(int j = 0; j < (int)nextPoints2f.size(); j++){
		    	nextOneKeyPoint.pt.x = nextPoints2f[j].x;
		    	nextOneKeyPoint.pt.y = nextPoints2f[j].y;
		    	if(minX > nextOneKeyPoint.pt.x){minX = (int)nextOneKeyPoint.pt.x;}
		    	if(maxX < nextOneKeyPoint.pt.x){maxX = (int)nextOneKeyPoint.pt.x;}
		    	if(minY > nextOneKeyPoint.pt.y){minY = (int)nextOneKeyPoint.pt.y;}
		    	if(maxY < nextOneKeyPoint.pt.y){maxY = (int)nextOneKeyPoint.pt.y;}
		    	nextKeypoints.push_back(nextOneKeyPoint); 
				allNextKeypoints[PointNumsTillThisBbox[boxIdx] + j] = nextKeypoints[j];
				//PointIds.push_back(boxIdx);  should remain unchanged
			}
			
			cout << "ORIGINAL Bbox data" << endl; 
			cout << "x = " << Bboxes[BboxNum].x << ", y = " << Bboxes[BboxNum].y << ", x+width = " << Bboxes[BboxNum].x + Bboxes[BboxNum].width << ", y+height = " << Bboxes[BboxNum].y + Bboxes[BboxNum].height << endl; 
			
			Bboxes[BboxNum].x = minX;
			Bboxes[BboxNum].y = minY;
			Bboxes[BboxNum].width = maxX-minX;
			Bboxes[BboxNum].height = maxY-minY;

			cout << "minX " << minX << ", minY " << minY << ", width " << maxX-minX << ", height " << maxY-minY << endl; 

			cout << "RENEWED Bbox data" << endl; 
			cout << "x = " << Bboxes[BboxNum].x << ", y = " << Bboxes[BboxNum].y << ", width = " << Bboxes[BboxNum].width << ", height = " << Bboxes[BboxNum].height << endl; 


			// Mat debug_frame2; 
	  //   	debug_frame2 = _frame.clone();
			// drawKeypoints(_frame, nextKeypoints, debug_frame2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			// rectangle(debug_frame2, Point(Bboxes[BboxNum].x, Bboxes[BboxNum].y), Point(Bboxes[BboxNum].x + Bboxes[BboxNum].width, Bboxes[BboxNum].y + Bboxes[BboxNum].height), Scalar(255,255,255)); 
			// imshow("debugFeatures after", debug_frame2);
			// getchar();

			cout << "-----------------" <<endl << "After SingleTracker on " << BboxNum << " Bbox, " << endl; 
			cout << "BoxId in this Bbox " << BoxIds[BoxIds.size()-1] << endl; 
			cout << "PointNumsTillThisBbox has size " << PointNumsTillThisBbox.size() << endl; 
			cout << "The current element of PointNumsTillThisBbox is " << PointNumsTillThisBbox[BboxNum] << endl; 
			cout << "PointIds(all the same) in this Bbox " << PointIds[(int)( allNextKeypoints.size()-nextKeypoints.size() )] << endl; 
			cout << "All together #PointIds add in this Bbox(PointNums) " << PointNums[BboxNum] << endl;
			cout << "Bbox width after max, min is " << Bboxes[BboxNum].width << endl;  
			cout << "-----------------" << endl << "SingleTracker ended this time" << endl << "--------------" << endl; 
			// substitute Keypoints with nextKeypoints
			// Keypoints.clear();
			// for(int j = 0; j < (int)nextKeypoints.size(); j++){
			// 	Keypoints.push_back(nextKeypoints[j]); 
			// }
		}

		
	}
	/////addDetection should rearrange allNextKeypoints
	void addDetection(Mat _oldframe, Mat _frame,Rect _bboxes){ //seems tracker in CV is a function
		//assume bboxes are already there
		//int nextId = 0; 
		cout << "======================"<<"start addDectection" <<endl;
		Mat oldframe_gray; 
		cvtColor( _oldframe, oldframe_gray, CV_BGR2GRAY );
	    equalizeHist( oldframe_gray, oldframe_gray );
	    // prepare frame_gray
	    Mat frame_gray;
		cvtColor( _frame, frame_gray, CV_BGR2GRAY );
	    equalizeHist( frame_gray, frame_gray );

    	Keypoints.clear();
    	currentPoints2f.clear(); 
	    nextPoints2f.clear();
	    nextKeypoints.clear();

		
			int boxIdx = findMatchingBox(_bboxes); 

			if (boxIdx == NOTHING){
				Bboxes.push_back(_bboxes);
				cout << "push_back a box " << endl; 
				cout << "BOX_w = " << _bboxes.width << endl; 
				BoxScores.push_back(1); 
				 

				Mat Mask = Mat::zeros(_oldframe.size(), CV_8U); 
			    Mat ROI(Mask, _bboxes);// init the mask matrix
			    ROI = Scalar(255,255,255);
			    double MinHessian = 400;
			    int octaves = 3;
			    int octaveLayers = 6;
			    SurfFeatureDetector sDetector(MinHessian, octaves, octaveLayers);
				sDetector.detect(oldframe_gray, Keypoints, Mask);

				for(int k = 0; k < (int)Keypoints.size(); k++){ 
					allNextKeypoints.push_back(Keypoints[k]);
					PointIds.push_back(nextId); 
				}
				BoxIds.push_back(nextId);
				PointNums.push_back((int)Keypoints.size());
				
				// construct PointNumsTillThisBbox in this way to make it 1 longer that PointNums 
				if( (int)PointNums.size() == 1){
					cout << "1 case" << endl; 
					PointNumsTillThisBbox.push_back( 0 );
					PointNumsTillThisBbox.push_back( PointNums[0] );
				}else{
					cout << "end case" << endl; 
					PointNumsTillThisBbox.push_back( PointNumsTillThisBbox[(int)PointNums.size()-1] + PointNums[(int)PointNums.size()-1] );
					}

				
				nextId++;
				//flagForAddDetection = 1; 
			}
			else{
				cout << "Calling deleteBox() " << endl;				
				int currentScore = deleteBox(boxIdx);
				cout << "deleteBox() finished" << endl;
				
				Bboxes.push_back(_bboxes); 
				BoxScores.push_back(currentScore + 1); 
				

				// find feature points to update other variables
				
				Mat Mask = Mat::zeros(_frame.size(), CV_8U); 
			    Mat ROI(Mask, _bboxes);// init the mask matrix
			    ROI = Scalar(255,255,255);
			   
			    double MinHessian = 400;
			    int octaves = 3;
			    int octaveLayers = 6;
			    SurfFeatureDetector sDetector(MinHessian, octaves, octaveLayers);
				sDetector.detect(frame_gray, Keypoints, Mask);
				for(int k = 0; k < (int)Keypoints.size(); k++){ 
					allNextKeypoints.push_back(Keypoints[k]);
					PointIds.push_back(boxIdx); 
				}
				BoxIds.push_back(boxIdx);
				PointNums.push_back((int)Keypoints.size());
				// make PointNumsTillThisBox 1 longer
				PointNumsTillThisBbox.push_back(PointNumsTillThisBbox[PointNumsTillThisBbox.size()-1] + (int)Keypoints.size()); 
				
			}
		
		
	}

	int deleteBox(int _boxIdx){
		cout << "---==---" << endl;
		vector<int> indice; 
		indice = findIndices(BoxIds, _boxIdx); 
		int _currentScore;
		cout << "Bboxes's Id before deleteBox" <<endl; 
		for(int k = 0; k < (int)Bboxes.size(); k++){
			cout << BoxIds[k] << endl;
		}
		cout << "Bboxes's scores " << endl; 
		for(int k = 0; k < (int)Bboxes.size(); k++){
			cout << BoxScores[k] << endl;
		}
		cout << "deleting indices has size " << indice.size() << endl;
		
		for(int j = 0; j < (int)indice.size(); j++){ // actually size will only be 1 in this case
			cout << "indice number " << indice[0] << endl;
			cout << "BboxesNum " << Bboxes.size() << endl;
			Bboxes.erase(Bboxes.begin() + indice[j]); 
			cout << "BboxesNum after erase " << Bboxes.size() << endl; 
			_currentScore = BoxScores[indice[j]]; 
			BoxScores.erase(BoxScores.begin() + indice[j]); 
			BoxIds.erase(BoxIds.begin() + indice[j]);
			PointNums.erase(PointNums.begin() + indice[j]);
			for(int k = indice[j]+1; k < (int)PointNumsTillThisBbox.size()-1; k++){
				PointNumsTillThisBbox[k] = PointNumsTillThisBbox[k-1] + PointNumsTillThisBbox[k+1] - PointNumsTillThisBbox[k];
			}
			
			PointNumsTillThisBbox.erase(PointNumsTillThisBbox.begin() + PointNumsTillThisBbox.size() - 1);
			cout << "464" << endl;
		}
			 
		vector<int> indices; 
		cout << "_boxIdx = " << _boxIdx << endl;
		indices = findIndices(PointIds, _boxIdx); 
		cout << "indices.size() = " << indices.size() << endl;
		cout << "PointIds START +++++++++++++++++++++" << endl;
		// for(int jj = 0; jj < (int)PointIds.size(); jj++){
			
		// 	cout << PointIds[jj] << " ";

		// }
		cout << endl << "PointIds END +++++++++++++++++++++" << endl;

		cout << "indices START +++++++++++++++++++++" << endl;
		// for(int jj = 0; jj < (int)indices.size(); jj++){
			
		// 	cout << indices[jj] << " ";

		// }
		cout << endl << "indices END +++++++++++++++++++++" << endl;
		for(int k = 0; k < (int)indices.size(); k++){
			PointIds.erase(PointIds.begin() + indices[0]); 
			allNextKeypoints.erase(allNextKeypoints.begin() + indices[0]); 
		}
		
		cout << "AFTER PointIds START +++++++++++++++++++++" << endl;
		// for(int jj = 0; jj < (int)PointIds.size(); jj++){
			
		// 	cout << PointIds[jj] << " ";

		// }
		cout << endl << "AFTER PointIds END +++++++++++++++++++++" << endl;

		cout <<"---==---" << endl;
		cout << "Bboxes's Id after deleteBox" <<endl; 
		// for(int k = 0; k < (int)Bboxes.size(); k++){
		// 	cout << BoxIds[k] << endl;
		// }
	
		return _currentScore; 
	}
	vector<int> findIndices(vector<int> BoxIdsOrPointIds, int _boxIdx){
		vector<int> _indices; 
		for(int j = 0; j < (int)BoxIdsOrPointIds.size(); j++){
			if((int)BoxIdsOrPointIds[j] == _boxIdx){
				_indices.push_back(j); 
			}
		}
		return _indices; 
	}
	int findMatchingBox(Rect box){
		for(int i = 0; i < (int)Bboxes.size(); i++){
			////
			area = computeRectJoinUnion(Bboxes[i], box);
			if(area > AREA_RATIO * Bboxes[i].width * Bboxes[i].height){
				BoxIdx = BoxIds[i]; // Here's the problem
				//BoxIdx = 111; 
				cout << "(findMatchingBox) BoxIdx returned " << BoxIdx << endl; 
				return BoxIdx; 
			}
		}
		cout << "(findMatchingBox) BoxIdx returned " << NOTHING << endl;
		return NOTHING; 
	}

private:
	
	

	/** @function detectAndDisplay */
	void detectAndDisplay( Mat _frame ) // currently the old algorithm, using red rects to show tracked tools
	{
	    std::vector<Rect> tools;

	    Mat frame_gray;

	    cvtColor( _frame, frame_gray, CV_BGR2GRAY );
	    equalizeHist( frame_gray, frame_gray );
	    //-- Detect, and save results in "tools"
	    casClassifier.detectMultiScale( frame_gray, tools, 1.005, 2, 0|CASCADE_SCALE_IMAGE, Size(0, 0) );

	    for( size_t i = 0; i < tools.size(); i++ ){
	  	    
	  	    rectangle(_frame, Point(tools[i].x, tools[i].y), Point(tools[i].x + tools[i].width, tools[i].y + tools[i].height), Scalar(255,255,255)); 

	  	    Mat toolROI = frame_gray( tools[i] );
	  	    
	  	    //printf("Tool detected in this frame. i = %d\n", (int)i + 1);
	  	    
	  //       if(loop%10 == 0 ){
			// 	
		 //  	    flag = 1; 	  	    
			// } 
	        _lasttools = tools; 
	        bboxes = tools;//In vector, = is copy
	    }
	    // if(flag  == 1){
	    _capturedtools.clear(); 
    	_capturedtools = Bboxes; 
		for( int i = 0; i < (int)_capturedtools.size(); i++){
			rectangle(cap_frame, Point(_capturedtools[i].x, _capturedtools[i].y), Point(_capturedtools[i].x + _capturedtools[i].width, _capturedtools[i].y + _capturedtools[i].height), Scalar(0,0,255));
		}
		// }
	    //-- Show what you got
	    imshow( window_name, _frame );
	    imshow( "captured", cap_frame); 
	 }


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
			newBboxes[u] = bboxes[Indexes[u]];
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
		//printf("In the tracking mode, ratio = %f\n", (float)(AJoin * AJoin)/(A1*A2));
	        return AJoin;
		
		}                   
	    else{
		//printf("Nothing can be tracked in this frame! \n");
		//printf("abs_ratio = %f \n", edge_ratio); 
	        return 0;
		}
	}

};