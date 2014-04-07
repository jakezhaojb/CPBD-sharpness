#ifndef CPDB_COMPUTE_H_
#define CPDB_COMPUTE_H_

#include <iostream>
#include <string.h>
#include <cmath>
#include <math.h>
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <armadillo>  
#include "time.h"  

using namespace std;
using namespace cv;
using namespace arma;


arma::mat cv_img2arma_mat(IplImage* img);
void edge_sobel(IplImage* image_origin,IplImage* sobelall);
void _AdaptiveFindThreshold(IplImage *dx, CvMat *dy, double &low, double &high);
void AdaptiveFindThreshold(const IplImage* image, double &low, double &high, int aperture_size=3);
double get_contrast_block(IplImage* gray_region_Img);
bool get_edge_blk_decision(IplImage* canny_region_Img,double threshold_);
arma::mat gradientY(arma::mat gray_image);
arma::mat gradientX(arma::mat gray_image);
void marziliano_method(IplImage* sobelImg, IplImage* gray_img,IplImage* width);
double cpdbm(IplImage* gray_img);

#endif
