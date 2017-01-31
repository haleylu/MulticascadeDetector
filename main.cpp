#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "MulticascadeDetector.cpp"


using namespace std;
using namespace cv;


int main( int argc, const char** argv ){
	MulticascadeDetector d; 
	d.detectProcess();
	
	// vector< vector <int> > vec;
	// vector<int> coords(4,7);
	// for(int i = 0; i < 10; i++){
	// 	vec.push_back(coords);
	// }
	
	// //cout << vec[1][1] << "\n";
	// cout << "vec size is" << vec.size() << "\n"; 

	 // MatrixXd m;
	 // cout << m.cols() << " " << m.rows(); 
	 //  m(0,0) = 3;
	 //  m(1,0) = 2.5;
	 //  m(0,1) = -1;
	 //  m(1,1) = m(1,0) + m(0,1);
	 //  std::cout << m << std::endl;

	vector< vector<float> > Points;
	for(int u = 0; u < 2000; u++){
	    	Points.push_back(vector<float>(2,2.f)); 
	    }
    cout << Points[1999][0] << endl << Points[20][1] << endl; 
    cout << "40" << endl; 
    Points.clear();
    cout << Points.size() << endl;

	return 0;
}
