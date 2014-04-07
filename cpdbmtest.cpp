#include <iostream>
#include <string.h>
#include <cmath>
#include <math.h>
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <armadillo>  
#include "time.h"  


#define PI        3.14159265358979323846
#define threshold 0.002
#define beta      3.6
#define block_num 10

using namespace std;
using namespace cv;
using namespace arma;

///**
// * Perform one thinning iteration.
// * Normally you wouldn't call this function directly from your code.
// *
// * @param  im    Binary image with range = 0-1
// * @param  iter  0=even, 1=odd
// */
//void thinningIteration(cv::Mat& im, int iter)
//{
//    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);
//
//    for (int i = 1; i < im.rows-1; i++)
//    {
//        for (int j = 1; j < im.cols-1; j++)
//        {
//            uchar p2 = im.at<uchar>(i-1, j);
//            uchar p3 = im.at<uchar>(i-1, j+1);
//            uchar p4 = im.at<uchar>(i, j+1);
//            uchar p5 = im.at<uchar>(i+1, j+1);
//            uchar p6 = im.at<uchar>(i+1, j);
//            uchar p7 = im.at<uchar>(i+1, j-1);
//            uchar p8 = im.at<uchar>(i, j-1);
//            uchar p9 = im.at<uchar>(i-1, j-1);
//
//            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
//                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
//                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
//                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
//            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
//            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
//            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
//
//            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
//                marker.at<uchar>(i,j) = 1;
//        }
//    }
//
//    im &= ~marker;
//}
//
///**
// * Function for thinning the given binary image
// *
// * @param  im  Binary image with range = 0-255
// */
//void thinning(IplImage* im1)
//{
//	cv::Mat im(im1); 
//	im /= 255;
//
//    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
//    cv::Mat diff;
//
//    do {
//        thinningIteration(im, 0);
//        thinningIteration(im, 1);
//        cv::absdiff(im, prev, diff);
//        im.copyTo(prev);
//    } 
//    while (cv::countNonZero(diff) > 0);
//
//    im *= 255;
//	im1= &IplImage(im);
//}

//opencv图像->arma::mat
arma::mat cv_img2arma_mat(IplImage* img,int M,int N)
{
	arma::mat dst;
	cv::Mat sobelImg_mat(img);
	cv::Mat_<uchar> img_ = img;
	dst.zeros(M,N);
	for(int i=0;i<M;i++)
	{
		for(int j=0;j<N;j++)
		{
			dst(i,j)=img_(i,j);
		}
	}
	return dst;
}

//matlab->edge with sobel
void edge_sobel(IplImage* image_origin,IplImage* sobelall)
{
	IplImage* image=cvCreateImage(cvSize(image_origin->width+2, image_origin->height+2),IPL_DEPTH_8U, 1);
	CvRect roi_rect;
	roi_rect.x=1;
	roi_rect.y=1;
	roi_rect.width=image_origin->width;
	roi_rect.height=image_origin->height;
	cvSetImageROI(image,roi_rect);
	cvCopy(image_origin,image);
	cvResetImageROI(image);
	//CvScalar a;
	//a=cvGet2D(image,1,1); 
	//cvSet2D(image,0,0,a); 
	//a=cvGet2D(image,1,image_origin->width); 
	//cvSet2D(image,0,image_origin->width+1,a); 
	//a=cvGet2D(image,image_origin->height,1); 
	//cvSet2D(image,image_origin->height+1,0,a); 
	//a=cvGet2D(image,image_origin->height,image_origin->width); 
	//cvSet2D(image,image_origin->height+1,image_origin->width+1,a); 
	//double time,time2;
	//clock_t start,finish,mid;
	//start=clock();
	//for(int i=1;i<image->height-1;i++)
	//{
	//	CvScalar a,b;
	//	a=cvGet2D(image,i,1); 
	//	cvSet2D(image,i,0,a);		
	//	b=cvGet2D(image,i,image->width-2); 
	//	cvSet2D(image,i,image->width-1,b);	
	//}
	for(int j=1;j<image->width-1;j++)
	{
		CvScalar a,b;
		a=cvGet2D(image,1,j); 
		cvSet2D(image,0,j,a);		
		b=cvGet2D(image,image->height-2,j); 
		cvSet2D(image,image->height-1,j,b);	
	}
	IplImage* sobelgx=cvCreateImage(cvSize(image_origin->width, image_origin->height),IPL_DEPTH_16S, 1);
	IplImage* sobelgy=cvCreateImage(cvSize(image_origin->width, image_origin->height),IPL_DEPTH_16S, 1);
	for(int i=1;i<image->height-1;i++) 
	{ 
		for(int j=1;j<image->width-1;j++) 
		{
			CvScalar d;
			if(j==1||j==image->width-2)
			{
				d.val[0]=0;
				cvSet2D(sobelgx,i-1,j-1,d); 
			}
			else
			{
			CvScalar s1,s2,s3,s4,s5,s6,s7,s8,s9; 
			s1 =cvGet2D(image,i-1,j-1); 
			s2 =cvGet2D(image,i-1,j); 
			s3 =cvGet2D(image,i-1,j+1); 
			s4 =cvGet2D(image,i,j-1); 
			s5 =cvGet2D(image,i,j); 
			s6 =cvGet2D(image,i,j+1); 
			s7 =cvGet2D(image,i+1,j-1); 
			s8 =cvGet2D(image,i+1,j); 
			s9 =cvGet2D(image,i+1,j+1); 

			CvScalar sx; 
			//CvScalar sy; 

			sx.val[0] =(s1.val[0] - s3.val[0] + 2*(s4.val[0] - s6.val[0]) + s7.val[0] - s9.val[0])/8; 
			//sy.val[0] =(s1.val[0] - s7.val[0] + 2*(s2.val[0] - s8.val[0]) + s3.val[0] - s9.val[0])/8; 

			cvSet2D(sobelgx,i-1,j-1,sx); 
			//cvSet2D(sobelgy,i-1,j-1,sy); 
			}
		} 
	} 
	//mid=clock();
	//time=double(mid-start);
	//std::cout<<"sobel_1step="<<time<<"ms"<<endl;
	IplImage* sobelall_16S=cvCreateImage(cvSize(image_origin->width, image_origin->height),IPL_DEPTH_16S, 1);
	cvCopy(sobelgx,sobelall_16S);
	for(int i=0;i<sobelall_16S->height;i++)
	{
		for(int j=0;j<sobelall_16S->width;j++)
		{
			CvScalar c;
			c =cvGet2D(sobelall_16S,i,j); 
			//if(i==0||i==sobelall_16S->height-1)
			//{
			//	c.val[0]=0;
			//	cvSet2D(sobelall,i,j,c); 
			//}
			//else
			if(j!=0&&j!=sobelall_16S->width-1)
			{
				CvScalar c1,c2;
				c1 =cvGet2D(sobelall_16S,i,j-1); 
				c2 =cvGet2D(sobelall_16S,i,j+1); 
				if(c.val[0]>=2||c.val[0]<=-2)
				{
					if(abs(c.val[0])>abs(c1.val[0]))
					{
						if(abs(c.val[0])>=abs(c2.val[0]))
						{
							c.val[0]=255;
							cvSet2D(sobelall,i,j,c); 
						}
					}
					
				}
				else
				{
					c.val[0]=0;
					cvSet2D(sobelall,i,j,c); 
				}
			}
		}
	}
	//finish=clock();
	//time2=double(finish-mid);
	//std::cout<<"sobel_1step="<<time2<<"ms"<<endl;
}

// 仿照matlab，自适应求高低两个门限                                            
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)   
{                                                                              
	CvSize size;                                                           
	IplImage *imge=0;                                                      
	int i,j;                                                               
	CvHistogram *hist;                                                     
	int hist_size = 255;                                                   
	float range_0[]={0,256};                                               
	float* ranges[] = { range_0 };                                         
	double PercentOfPixelsNotEdges = 0.7;                                  
	size = cvGetSize(dx);                                                  
	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);                          
	// 计算边缘的强度, 并存于图像中                                        
	float maxv = 0;                                                        
	for(i = 0; i < size.height; i++ )                                      
	{                                                                      
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);        
		const short* _dy = (short*)(dy->data.ptr + dy->step*i);        
		float* _image = (float *)(imge->imageData + imge->widthStep*i);
		for(j = 0; j < size.width; j++)                                
		{                                                              
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));        
			maxv = maxv < _image[j] ? _image[j]: maxv;             
	                                                                       
		}                                                              
	}                                                                      
	if(maxv == 0){                                                         
		*high = 0;                                                     
		*low = 0;                                                      
		cvReleaseImage( &imge );                                       
		return;                                                        
	}                                                                      
                                                                               
	// 计算直方图                                                          
	range_0[1] = maxv;                                                     
	hist_size = (int)(hist_size > maxv ? maxv:hist_size);                  
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);          
	cvCalcHist( &imge, hist, 0, NULL );                                    
	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges); 
	float sum=0;                                                           
	int icount = hist->mat.dim[0].size;                                    
                                                                               
	float *h = (float*)cvPtr1D( hist->bins, 0 );                           
	for(i = 0; i < icount; i++)                                            
	{                                                                      
		sum += h[i];                                                   
		if( sum > total )                                              
			break;                                                 
	}                                                                      
	// 计算高低门限                                                        
	*high = (i+1) * maxv / hist_size ;                                     
	*low = *high * 0.4;                                                    
	cvReleaseImage( &imge );                                               
	cvReleaseHist(&hist);                                                  
}                                                                              
void AdaptiveFindThreshold(const CvArr* image, double *low, double *high, int aperture_size=3)
{                                                                              
	cv::Mat src = cv::cvarrToMat(image);                                   
	const int cn = src.channels();                                         
	cv::Mat dx(src.rows, src.cols, CV_16SC(cn));                           
	cv::Mat dy(src.rows, src.cols, CV_16SC(cn));                           
                                                                               
	cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, 4);
	cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, 4);
                                                                               
	CvMat _dx = dx, _dy = dy;                                              
	_AdaptiveFindThreshold(&_dx, &_dy, low, high);                         
                                                                               
}     

//function contrast = get_contrast_block(A)
double get_contrast_block(IplImage* gray_region_Img)
{
	double contrast;
	double MinValue;
	double MaxValue;
	cvMinMaxLoc(gray_region_Img,& MinValue,& MaxValue);
	contrast=MaxValue-MinValue;
	return contrast;
}

//function [im_out] = get_edge_blk_decision(im_in,T)
bool get_edge_blk_decision(IplImage* canny_region_Img,double threshold_)
{
	int nHeight=canny_region_Img->height;
	int nWidth=canny_region_Img->width;
	int L = nHeight*nWidth;
	double im_edge_pixels=0;
	bool im_out=false;
	cv::Mat input_imag(canny_region_Img);
	cv::Mat_<uchar> input_imag_ = input_imag;
	for(int i=0;i<nHeight;i++)
	{
		for(int j=0;j<nWidth;j++)
		{
			im_edge_pixels=im_edge_pixels+input_imag_(i,j)/255;
		}
	}
	im_out = im_edge_pixels > (L*threshold) ;
	return im_out;
}

//matlab->gradient函数
arma::mat gradientY(arma::mat gray_image)
{
	arma::mat gradientY_mat;
	gradientY_mat.copy_size( gray_image ) ;
	for(int i=0;i<(int)(gray_image.n_rows)-1;i++) 
	{ 
		if(i==0)
		{
			rowvec a = gray_image.row(i);
			rowvec b = gray_image.row(i+1);
			gradientY_mat.row(i)=b-a;
		}
		else if(i==gray_image.n_rows-1)
		{
			rowvec a = gray_image.row(i-1);
			rowvec b = gray_image.row(i);
			gradientY_mat.row(i)=b-a;
		}
		else
		{
			rowvec a = gray_image.row(i-1);
			rowvec b = gray_image.row(i+1);
			gradientY_mat.row(i)=(b-a)/2;
		}
	} 
	//cout<<gradientY_mat.max()<<" "<<gradientY_mat.min()<<endl;////无零值
	//for(int i=0;i<(int)(gray_image.n_rows)-1;i++) 
	//{
	//	for(int j=0;j<(int)(gray_image.n_cols)-1;j++) 
	//	{
	//		if(gradientY_mat(i,j)<-6.2e+066)
	//		{
	//			gradientY_mat(i,j)=0;
	//		}
	//	}
	//}
		
	return gradientY_mat;
}
arma::mat gradientX(arma::mat gray_image)
{
	arma::mat gradientX_mat;
	gradientX_mat.copy_size( gray_image ) ;
	for(int i=0;i<(int)(gray_image.n_cols)-1;i++) 
	{ 
		if(i==0)
		{
			colvec a = gray_image.col(i);
			colvec b = gray_image.col(i+1);
			gradientX_mat.col(i)=b-a;
		}
		else if(i==gray_image.n_cols-1)
		{
			colvec a = gray_image.col(i-1);
			colvec b = gray_image.col(i);
			gradientX_mat.col(i)=b-a;
		}
		else
		{
			colvec a = gray_image.col(i-1);
			colvec b = gray_image.col(i+1);
			gradientX_mat.col(i)=(b-a)/2;
		}
	} 
	return gradientX_mat;
}

//function [edge_width_map] = marziliano_method(E, A) 
void marziliano_method(IplImage* sobelImg, IplImage* gray_img,IplImage* width)
{
	int M, N;
	CvScalar width_count_side1,width_count_side2,width_count_side;
	width_count_side1.val[0]=0;
	width_count_side2.val[0]=0;
	M=gray_img->height;
	N=gray_img->width;
	arma::mat gray_image_mat;
	gray_image_mat=cv_img2arma_mat(gray_img,M,N);
	arma::mat grad_x_mat_;
	arma::mat grad_y_mat_;
	//**************
	//clock_t start, finish;   
	//double time_fun;
	//start=clock();
	///
	grad_x_mat_=gradientX(gray_image_mat);
	grad_y_mat_=gradientY(gray_image_mat);
	//**************
	//finish=clock();
	//time_fun=double(finish-start);
	//std::cout <<"gradient_time="<<time_fun <<"ms"<<std::endl;
	////
	arma::mat E;
	E=cv_img2arma_mat(sobelImg,M,N);
	arma::mat A;
	A=cv_img2arma_mat(gray_img,M,N);
	arma::mat angle_A;
	angle_A.zeros(M,N);	
	CvScalar s;
	s.val[0]=0;
	for(int a=0;a<M;a++)
	{
		for(int b=0;b<N;b++)
		{
			cvSet2D(width,a,b,s);//IplImage* width全零
		}
	}
	for(int m=0;m<M;m++)
	{
		for(int n=0;n<N;n++)
		{
			if(grad_x_mat_(m,n)!=0&&grad_x_mat_(m,n)>-6.2e+066)
			{
				double y=grad_y_mat_(m,n);
				double x=grad_x_mat_(m,n);
				angle_A(m,n) = atan2(y,x)*(180/PI);
			}
			if((grad_x_mat_(m,n)==0||grad_x_mat_(m,n)<-6.2e+066)&&(grad_y_mat_(m,n)==0||grad_y_mat_(m,n)<-6.2e+066))
			{
				angle_A(m,n) =0;
			}
			if((grad_x_mat_(m,n)==0||grad_x_mat_(m,n)==-6.27744e+066)&&grad_y_mat_(m,n)==PI/2)
			{
				angle_A(m,n) =90;
			}
		}
	}
	if(angle_A.n_elem!=0)
	{
		arma::mat angle_Arnd = 45*round(angle_A/45);
		for(int m=1;m<M-1;m++)
		{
			for(int n=1;n<N-1;n++)
			{
				if(E(m,n)==255)
				{
					if(angle_Arnd(m,n)==180||angle_Arnd(m,n)==-180)
					{
						int width_a=0,width_b=0;
						for(int k=0;k<=100;k++)
						{
							int posy1=n-k;
							int posy2=n-1-k;
							if(posy2<=0) 
							{
								width_a=k;
								break;
							}
							if((A(m,posy2-1)-A(m,posy1-1))<=0)
							{
								width_a=k;
								break;
							}
							
						}
						width_count_side1.val[0] =width_a + 1 ;
						for(int k=0;k<=100;k++)
						{
							int negy1=n+2+k;
							int negy2=n+3+k;
							if(negy2>N)
							{
								width_b=k;
								break;
							}
							if((A(m,negy2-1)-A(m,negy1-1))>=0)
							{
								width_b=k;
								break;
							}
							
						}
						width_count_side2.val[0] = width_b+1 ;
						width_count_side.val[0]=width_count_side1.val[0]+width_count_side2.val[0];
						cvSet2D(width,m,n,width_count_side);
					}
					if(angle_Arnd(m,n)==0)
					{
						int width_a=0,width_b=0;
						for(int k=0;k<=100;k++)
						{
							int posy1=n+2+k;
							int posy2=n+3+k;
							if(posy2>N) 
							{
								width_a=k;
								break;
							}
							if((A(m,posy2-1)-A(m,posy1-1))<=0)
							{
								width_a=k;
								break;
							}
							
						}
						width_count_side1.val[0] = width_a + 1 ;
						for(int k=0;k<=100;k++)
						{
							int negy1=n-k;
							int negy2=n-1-k;
							if(negy2<=0)
							{
								width_b=k;
								break;
							}
							if((A(m,negy2-1)-A(m,negy1-1))>=0)
							{
								width_b=k;
								break;
							}
							
						}
						width_count_side2.val[0] = width_b + 1 ;
						width_count_side.val[0]=width_count_side1.val[0]+width_count_side2.val[0];
						cvSet2D(width,m,n,width_count_side);
					}
				}
			}
		}
	}
}

//function [sharpness_metric] = CPBD_compute(input_image)
double cpdbm(IplImage* gray_img)
{
	double cpdbm_value=0;
	double blk_jnb;
	int total_num_edges=0;
	int temp_index;
	int nHeight, nWidth;
	bool decision;
	double contrast;
	arma::mat hist_pblur;
	hist_pblur.zeros(1,101);
	cv::Mat widthjnb_1 = cv::Mat::ones(1, 51, CV_8U)*5;
	cv::Mat widthjnb_2 = cv::Mat::ones(1, 205, CV_8U)*3;
	cv::Mat widthjnb= cv::Mat::ones(1, 256, CV_8U);
	cv::Mat_<uchar> widthjnb_1_ = widthjnb_1;
	cv::Mat_<uchar> widthjnb_2_ = widthjnb_2;
	cv::Mat_<uchar> widthjnb_ = widthjnb;
	for(int i=0;i<51;i++)
	{
		widthjnb_(0,i)=widthjnb_1_(0,i);
	}
	for(int i=0;i<205;i++)
	{
		widthjnb_(0,51+i)=widthjnb_2_(0,i);
	}
	nHeight=gray_img->height;
	nWidth=gray_img->width;
	int max_sq=nHeight>nWidth?nHeight:nWidth;
	int rb=max_sq/block_num;
	int rc=rb;
	IplImage* sobelImg  = cvCreateImage(cvSize(nWidth, nHeight),IPL_DEPTH_8U, 1);
    IplImage* cannyImg  = cvCreateImage(cvSize(nWidth, nHeight),IPL_DEPTH_8U, 1);
	IplImage* smooth_dst  = cvCreateImage(cvSize(nWidth, nHeight),IPL_DEPTH_8U, 1);
	///************
	//clock_t start, finish,mid;   
	//double time_fun,time_mid;
	//start=clock();
	///
	double low = 0.0, high = 0.0;
	edge_sobel(gray_img,sobelImg);
	//cvShowImage("edge_sobel",sobelImg);
	///**********
	//mid=clock();
	//time_mid=double(mid-start);
	//std::cout <<"edge_sobel_time="<<time_mid <<"ms"<<std::endl;
	////
	cvSmooth(gray_img,smooth_dst,CV_GAUSSIAN,5,5,2);
	AdaptiveFindThreshold(smooth_dst, &low, &high);
	cvCanny(smooth_dst, cannyImg, low, high);
	//thinning(cannyImg);//////////////////////
	//cvShowImage("cannyImg",cannyImg);
	///***************
	//finish=clock();
	//time_fun=double(finish-mid);
	//std::cout <<"edge_canny_time="<<time_fun <<"ms"<<std::endl;
	///
	IplImage* cannyImg_1;
	IplImage* edge_width_map_1;
	IplImage* gray_img_1;
	gray_img_1=cvCreateImage(cvSize(rb,rc),gray_img->depth,gray_img->nChannels);
	edge_width_map_1 =cvCreateImage(cvSize(rb,rc),cannyImg->depth,cannyImg->nChannels);
	cannyImg_1=cvCreateImage(cvSize(rb,rc),cannyImg->depth,cannyImg->nChannels);
	IplImage* edge_width_map;
	edge_width_map=cvCreateImage(cvGetSize(gray_img),gray_img->depth,gray_img->nChannels);
	marziliano_method(sobelImg, gray_img,edge_width_map);
	for(int p=0;p<=nHeight-rb;p=p+rb)
	{
		for(int q=0;q<=nWidth-rc;q=q+rc)
		{
			CvRect roi_rect;
			roi_rect.x=q;
			roi_rect.y=p;
			roi_rect.width=rb;
			roi_rect.height=rc;
			cvSetImageROI(cannyImg,roi_rect);
			cvCopy(cannyImg,cannyImg_1);
			cvSetImageROI(edge_width_map,roi_rect);
			cvCopy(edge_width_map,edge_width_map_1);
			cvSetImageROI(gray_img,roi_rect);
			cvCopy(gray_img,gray_img_1);
			decision = get_edge_blk_decision(cannyImg_1, threshold);
			cvResetImageROI(cannyImg);
			if(decision==true)
			{
				contrast = get_contrast_block(gray_img_1)+1;
				blk_jnb = widthjnb_(0,contrast-1);
				for(int m=0;m<rb;m++)
				{
					for(int n=0;n<rc;n++)
					{
						CvScalar a;
						a=cvGet2D(edge_width_map_1,m,n); 
						if(a.val[0]!=0)
						{
							double local_width_d=a.val[0];
							double prob_blur_detection = 1 - exp(-pow(abs(local_width_d/blk_jnb),beta));
							temp_index = (int)(prob_blur_detection* 100+0.5) + 1;
							hist_pblur(0,temp_index-1) = hist_pblur(0,temp_index-1) + 1;
							total_num_edges = total_num_edges + 1;
						}
					}
				}
			}
			cvResetImageROI(edge_width_map);
			cvResetImageROI(gray_img_1);
		}
	}
	 if(total_num_edges !=0)
	 {
		 hist_pblur = hist_pblur / total_num_edges;
	 }
	 else
	 {
		 hist_pblur.zeros(1,101);
	 }
	 cpdbm_value=accu(hist_pblur.cols(0,63));
	 return cpdbm_value;
}

//main
int main(int argc, char** argv)

{
	clock_t start, finish;   
	double time_fun;
	IplImage* img ;
	IplImage* gray;
	//cvCvtColor(img,gray,CV_BGR2GRAY);
	//cvNamedWindow("RGB");;
	//cvShowImage("RGB",img);
	//start=clock();
	//double cpdbm_value=cpdbm(gray);
	//finish=clock();
	//time_fun=double(finish-start);
	//std::cout << "cpdbm_value="<<cpdbm_value <<std::endl;
	//std::cout <<"total_time="<<time_fun <<"ms"<<std::endl;
	//cvWaitKey(0);
    //cvDestroyWindow("RGB");	
	
	fstream fout;
	fout.open("output1.txt"); 
	char filename[20];
	for(int i=0; i<535; i++) 
	{ 
		sprintf(filename, ".//tp1//m (%d).jpg", i+1); 
		img=cvLoadImage(filename,1);
	    gray = cvCreateImage(cvGetSize(img),img->depth,1);
		//cvShowImage("RGB",img);
		//cvShowImage("gray",gray);
		cvCvtColor(img,gray,CV_BGR2GRAY);
	    start=clock();
		double cpdbm_value=cpdbm(gray);
		finish=clock();
		time_fun=double(finish-start); 
		fout << filename << "  "<<cpdbm_value<<"  "<< time_fun<< "ms"<<"\n";
    }  
	fout << flush; fout.close();
}