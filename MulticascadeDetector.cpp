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
	int flag;

	// MATLAB re-creating
	vector< Rect > Bboxes; 
	vector<Rect> bboxes; 
	vector<int> BoxIds; 
	vector<vector <int> > Points;
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
    // Size WinSize;
    // int maxLevel;
    // TermCriteria termcrit;
    // int flags;
    // double minEigThreshold;
    vector< vector<int> > currentPoints; 
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
		flag = 0;
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
		std::vector<int> BoxIds(500); 
		vector<vector <int> > Points(500, vector<int>(4));
		std::vector<int> PointIds(500,8);
		std::vector<int> BoxScores(500); 
		nextId = 1;
		vector<vector <int> > bbox1(500, std::vector<int>(4)); 
		area = 0; 
		BoxIdx = 0; 

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
    	// for( int u = 0; u < 500; u++){
    	// 	nextKeypoints.push_back(keyPointIniter);
    	// }
    	
    	nextOneKeyPoint.pt.x = 0;
    	nextOneKeyPoint.pt.y = 0; 

    	for( int u = 0; u < 2000; u++){
    		//allNextKeypoints.push_back(keyPointIniter);
    	}
	}

	void initDetector(){
		vector <float> err(100);
	    vector <uchar> Status(100);
	    // Size WinSize = Size(31,31);
	    // int maxLevel = 3;
	    // TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.03);
	    // int flags = 0;
	    // double minEigThreshold = 1e-4;
	}


	void detectProcess(){
 		
 		//settings for detect
		int i = 0; 
		Mat frame;
		Mat oldframe;
		

		if( !casClassifier.load( cascade_name ) ){ printf("--(!)Error loading\n");};

		cout << "READ IN FRAMES STARTS" << endl; 
	        VideoCapture cap(video_name); 
	        if( !cap.isOpened() ){
		     printf("cap is not opened");
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, 10);


		cap >> frame; 
		cap >> cap_frame;

			while( 1 ){
			cout << endl << endl; 
			printf("The %dth frame started. \n", i); 
	    	i++;

			if( !frame.empty() ){ 
			    loop++; 
			    // Detect bounding boxes in the frame, save the corrdinates in Bboxes
			    // then display the Bboxes
			    detectAndDisplay( frame ); 
			    // Rearrange Bboxes by left top point.x
			    sortedBboxesId = sortRectByXAndGiveBackIndexes(bboxes); 
			    
			    bboxes = rearrangeBboxesUsingSortedIndexes(sortedBboxesId); 
			    
			    
			}else{ 
			    printf(" --(!) No captured frame -- Break! \n"); break; 
			}
			oldframe = frame.clone(); 
			cap >> frame; 
	    	cap >> cap_frame;
			
	    	cout << "All togeether " << bboxes.size() << " bboxes" << endl; 
		    // Find(only once) and track the feature points in two consecutive frames
			// member "nextPoints2f" is tracked points
			cout << "-------------------" << endl; 

			for(int j = 0; j < (int)bboxes.size(); j++){
				addDetection(oldframe, frame, bboxes[j]); 
				cout << "After addDetection(), Bboxes has size = " << Bboxes.size() << endl;
				cout << "--------------------" << endl;
			}
			for(int j = 0; j < (int)Bboxes.size(); j++){
				SingleTracker(oldframe, frame, j);
			}
		 	
			// draw all the point at a time
			Mat nextKey_frame; 
			drawKeypoints(frame, allNextKeypoints, nextKey_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			imshow("nextFeatures", nextKey_frame);

			for(int j = 0; j < (int)Bboxes.size(); j++){

			}
			
			for( int j = 0; j < (int)Bboxes.size(); j++){
				cout << "the " << j << "th " << "Bboxes, " << ///
				"x= " << Bboxes[j].x << " y = " << Bboxes[j].y << endl;  
			}
			// to make sure it's tracking, not dupllicating
			// for( int j = 0; j < 10; j++){
			// 	cout << "currentPoints, " << j << " x = " << currentPoints2f[j].x << ///
			// 	" y = " << currentPoints2f[j].y << endl;
			// 	cout << "nextPoints, " << j << " x = " << nextPoints2f[j].x << ///
			// 	" y = " << nextPoints2f[j].y << endl;
			// }
			cout << "allNextKeypoints.size() = " << allNextKeypoints.size() << endl; 

			RedetectPointsFlag = 1; // close the Redetect status
			// if(i%10 == 0){
			// 	RedetectPointsFlag = 0; 
			// 	PointNumsTillThisBbox.clear();
		 //    	PointIds.clear(); 
		 //    	PointNums.clear();
		 //    	BoxIds.clear(); 
		 //    	allNextKeypoints.clear();
			// }

			int c = waitKey(10);
			if( (char)c == 'c' ) { break; }
      		} 
	}

	void SingleTracker(Mat _oldframe, Mat _frame, int BboxNum){
		
		if(RedetectPointsFlag == 0){
	    	// move this part to addDetection() 
	    	

	    }else{
	     	// prepare oldframe_gray
	     	cout << "250" << endl;
			Mat oldframe_gray; 
			cvtColor( _oldframe, oldframe_gray, CV_BGR2GRAY );
		    equalizeHist( oldframe_gray, oldframe_gray );
		    // prepare frame_gray
		    Mat frame_gray;
			cvtColor( _frame, frame_gray, CV_BGR2GRAY );
		    equalizeHist( frame_gray, frame_gray );
		    cout << "257" << endl;
	    	Keypoints.clear();
	    	currentPoints2f.clear(); 
		    nextPoints2f.clear();
		    nextKeypoints.clear();
	    	cout << "262" << endl;
		    //cout << "reached 266" << endl;
		    // if( (int)PointNumsTillThisBbox.size() < (BboxNum-1)){
		    // 	return; ///////okashi
		    // }
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
			
	    
	     
		    // PointNumsTillThisBbox.clear();
	    	// PointIds.clear(); 
	    	// PointNums.clear();
	    	// BoxIds.clear(); 
	    	// allNextKeypoints.clear();

		    // draw Keypoints in this Bbox in "Feature Points" window
		    cout << "295" <<endl;
			Mat Key_frame;
			drawKeypoints(_oldframe, Keypoints, Key_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			imshow("Feature Points", Key_frame); 
			cout << "299" << endl;
			// convert the Keypoints to Points2f for optical flow tracking
			KeyPoint::convert(Keypoints, currentPoints2f, pIDs);
			// optical flow, track currentPoints2f, give out nextPoints2f	    
			calcOpticalFlowPyrLK(oldframe_gray, frame_gray, currentPoints2f, nextPoints2f, Status, err);
		    
		    int boxIdx = BoxIds[BboxNum]; 
		    
		    // Rect _deletedBbox = Bboxes[BboxNum]; 
		     
		    int minX = 1000; 
		    int maxX = 0;
		    int minY = 1000;
		    int maxY = 0;
		    // update nextKeyPoints, allNextKeypoints, PointIds using nextPoints2f
		    for(int j = 0; j < (int)nextPoints2f.size(); j++){
		    	nextOneKeyPoint.pt.x = nextPoints2f[j].x;
		    	nextOneKeyPoint.pt.y = nextPoints2f[j].y;
		    	if(minX > nextOneKeyPoint.pt.x){minX = nextOneKeyPoint.pt.x;}
		    	if(maxX < nextOneKeyPoint.pt.x){maxX = nextOneKeyPoint.pt.x;}
		    	if(minY > nextOneKeyPoint.pt.y){minY = nextOneKeyPoint.pt.y;}
		    	if(maxY < nextOneKeyPoint.pt.y){maxY = nextOneKeyPoint.pt.y;}
		    	nextKeypoints.push_back(nextOneKeyPoint); 
				allNextKeypoints[PointNumsTillThisBbox[boxIdx] + j] = nextKeypoints[j];
				//PointIds.push_back(boxIdx);  should remain unchanged
			}
			Bboxes[BboxNum].x = minX;
			Bboxes[BboxNum].y = minY;
			Bboxes[BboxNum].width = maxX-minX;
			Bboxes[BboxNum].height = maxY-minY;
			//BoxIds.push_back(boxIdx);
			//PointNums.push_back((int)nextPoints2f.size());
			// construct PointNumsTillThisBbox in this way to make it 1 longer that PointNums 
			// if( (int)PointNums.size() == 1){
			// 	PointNumsTillThisBbox.push_back( 0 );
			// }else if((int)PointNums.size() == ((int)Bboxes.size())){
			// 	PointNumsTillThisBbox.push_back( PointNumsTillThisBbox[(int)PointNums.size()-2] + PointNums[(int)PointNums.size()-2] );
			// 	PointNumsTillThisBbox.push_back( PointNumsTillThisBbox[(int)PointNums.size()-1] + PointNums[(int)PointNums.size()-1] );
			// }else{
			// 	PointNumsTillThisBbox.push_back( PointNumsTillThisBbox[(int)PointNums.size()-2] + PointNums[(int)PointNums.size()-2] );
			// }
			//PointNumsTillThisBbox.push_back(PointNumsTillThisBbox[PointNumsTillThisBbox.size()-1] + (int)Keypoints.size()); 
			
			cout << "-----------------" <<endl << "After SingleTracker on " << BboxNum << " Bbox, " << endl; 
			cout << "BoxId in this Bbox " << BoxIds[BoxIds.size()-1] << endl; 
			cout << "PointNumsTillThisBbox has size " << PointNumsTillThisBbox.size() << endl; 
			cout << "The current element of PointNumsTillThisBbox is " << PointNumsTillThisBbox[BboxNum] << endl; 
			cout << "PointIds(all the same) in this Bbox " << PointIds[(int)( allNextKeypoints.size()-nextKeypoints.size() )] << endl; 
			cout << "All together #PointIds add in this Bbox(PointNums) " << PointNums[BboxNum] << endl;
			cout << "-----------------" << endl << "SingleTracker ended" << endl << "--------------" << endl; 
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
				cout << "426" << endl;
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
			if(area > 0.7 * Bboxes[i].width * Bboxes[i].height){
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
	    int k = 0; 
	    //-- Detect tools
	    casClassifier.detectMultiScale( frame_gray, tools, 1.005, 2, 0|CASCADE_SCALE_IMAGE, Size(0, 0) );

	    for( size_t i = 0; i < tools.size(); i++ ){
	  	    
	  	    rectangle(_frame, Point(tools[i].x, tools[i].y), Point(tools[i].x + tools[i].width, tools[i].y + tools[i].height), Scalar(255,255,255)); 

	  	    Mat toolROI = frame_gray( tools[i] );
	  	    
	  	    //printf("Tool detected in this frame. i = %d\n", (int)i + 1);
	  	    
	        if(loop%10 == 0 && computeRectJoinUnion(tools[i], _lasttools[i]) > 0.9 ){
			k++;
			_capturedtools = tools; 
	  	        //rectangle(cap_frame, Point(tools[i].x, tools[i].y), Point(tools[i].x + tools[i].width, tools[i].y + tools[i].height), Scalar(0,0,255)); 
			//printf("Tool TRACKING TRACKING in this frame. k = %d, tool number is %d\n", (int)k, (int)i + 1);
			flag = 1; 	  	    
			} 
		if(flag  == 1){
			rectangle(cap_frame, Point(_capturedtools[i].x, _capturedtools[i].y), Point(_capturedtools[i].x + _capturedtools[i].width, _capturedtools[i].y + _capturedtools[i].height), Scalar(0,0,255));
		}
			k = 0;
	        _lasttools = tools; 

	        bboxes = tools;//In vector, = is copy
	    	
	    	
	    }
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
		printf("Nothing can be tracked in this frame! \n");
		//printf("abs_ratio = %f \n", edge_ratio); 
	        return 0;
		}
	}

};