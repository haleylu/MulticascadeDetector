#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "MulticascadeDetector.cpp"

#include <Eigen/Dense>
using Eigen::MatrixXd;

using namespace std;
using namespace cv;


int main( int argc, const char** argv ){
	// MulticascadeDetector d; 
	// d.detectProcess();
	
	// vector< vector <int> > vec;
	// vector<int> coords(4,7);
	// for(int i = 0; i < 10; i++){
	// 	vec.push_back(coords);
	// }
	
	// //cout << vec[1][1] << "\n";
	// cout << "vec size is" << vec.size() << "\n"; 

	 MatrixXd m(2,2);
	  m(0,0) = 3;
	  m(1,0) = 2.5;
	  m(0,1) = -1;
	  m(1,1) = m(1,0) + m(0,1);
	  std::cout << m << std::endl;

	return 0;
}
