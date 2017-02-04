#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "MulticascadeDetector.cpp"

// #include "rt_nonfinite.h"
// #include "cfind.h"
// #include "cfind_terminate.h"
// #include "cfind_emxAPI.h"
// #include "cfind_initialize.h"
// #include <stddef.h>
// #include <stdlib.h>
// #include <string.h>
// #include "rtwtypes.h"
// #include "cfind_types.h"

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

	vector<int> pencil;
	for(int j = 0; j < 10; j++){
		pencil.push_back(j); 
	}
	cout << "SIZE" << pencil.size() << endl;
	for(int u = 0; u < (int)pencil.size(); u++){
		cout << pencil[u] << " "; 
	}
	cout << endl; 

	pencil.erase(pencil.begin()); 
	cout << "SIZE" << pencil.size() << endl;
	for(int u = 0; u < (int)pencil.size(); u++){
		cout << pencil[u] << " "; 
	}
	cout << endl;

	pencil.erase(pencil.begin()+1, pencil.begin()+3);
	cout << "SIZE" << pencil.size() << endl;
	for(int u = 0; u < (int)pencil.size(); u++){
		cout << pencil[u] << " "; 
	}
	cout << endl;

	// emxArray_real_T *indexes_ = NULL;
 //  	emxArray_int32_T *X = NULL;
 //  	int32_T pencil_[(int)pencil.size()];
 //  	int indexes[(int)pencil.size()];  
 //  	for(int u = 0; u < (int)pencil.size(); u ++){
 //  		pencil_[u] = (int32_T)pencil[u];
 //  	}
 //  	X = emxCreateWrapper_int32_T(pencil_, 1, (int)pencil.size());
	// cfind(X, 4, indexes_);
	// // for(int u = 0; u < indexes_->size[0]; u ++){
 // //  		 indexes[u] = indexes_->data[u];
 // //  	}
	// emxDestroyArray_int32_T(X); 

	// cout << indexes_->data[0] << endl; 
	// // cout << indexes[0] << endl; 


	
	return 0;
}
