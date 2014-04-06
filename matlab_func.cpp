#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace Eigen;

// Transform from matlab matrix to Eigen::MatrixXd
vector<double> mat_assign_vec(MatrixXd x){
  vector<double> obj;
  for (int i = 0; i < x.size(); i++) {
    obj.push_back(x(i));
  }
  return obj;
}

// Transfrom from std::vector to Eigen:VectorXd
VectorXd vec_assign_mat(vector<double> x, bool order){
  if (!order) {
    // Fortran order, ColVector
    VectorXd obj(x.size());
    for (int i = 0; i < x.size(); i++) {
      obj(i) = x[i];
    }
    x.
    return obj;
  }
  else{
    // C++ order, RowVector
    RowVectorXd obj(x.size());
    for (int i = 0; i < x.size(); i++) {
      obj(i) = x[i];
    }
    return obj;
  }
}

// Transform from std::vector to Eigen::MatrixXd
MatrixXd vec_assign_mat(vector<double> x, int rows, int cols, bool order){
  int i, j;
  MatrixXd obj(rows, cols);
  if (order) {
    // C++ order, row
    for (i = 0; i < cols; i++) 
      for (j = 0; j < rows; j++) 
        obj(i ,j) = x[i*cols+j];
  }
  else{
    // Fortran order, column
    for (i = 0; i < rows; i++) 
      for (j = 0; j < cols; j++) 
        obj(i ,j) = x[i*rows+j];
  }
  return obj;
}

// Eigen::MatrixXd sort without rows and cols! 
VectorXd one_dim_mat_sort(MatrixXd x, bool order){
  if (x.rows()!=1 && x.cols()!=1) {
    cout<<"Wrong input 1d_mat_sort function\n"<<endl;
    exit(-1);
  }
  VectorXd sorted_1d_mat;
  vector<double> obj = mat_assign_vec(x);
  sort(obj.begin(), obj.end());
  sorted_1d_mat = vec_assign_mat(obj, order);
  return sorted_1d_mat;
}

// Eigen::MatrixXd sort with rows and cols! 
MatrixXd mat_sort(MatrixXd x, bool order){
  MatrixXd obj(x.rows(), x.cols());
  int i, j;
  if (order) {
    // C++ order, by row
    for (i = 0; i < x.rows(); i++) {
      obj.row(i) = one_dim_mat_sort(x.row(i), 0);
    }
  }
  else{
    // Fortran order, by column
    for (i = 0; i < x.cols(); i++) {
      obj.col(i) = one_dim_mat_sort(x.col(i), 1);
    }
  }
  return obj;
}


// std::vector sort with indexes using std::sort
struct data{
    double n;
    int index;
    bool operator<(const data& rhs) const{
      return n < rhs.n;
    }
  };

void vector_sort_with_index(vector<double>& x, int* index){
  int i;
  vector<data> x_;
  for (i = 0; i < x.size(); i++) {
    data d = {x[i], i};
    x_.push_back(d);  
  }
  sort(x_.begin(), x_.end());
  for (i = 0; i < x.size(); i++) {
    index[i] = x_[i].index;
    x[i] = x_[i].n;
  }
}

// Get a "clean" version Eigen::VectorXd whose elements are all from raw Eigen::MatrixXd
// but no replicated ones!
VectorXd unique(MatrixXd x){
  x.resize(x.size(), 1);
  VectorXd sorted_x = mat_sort(x, 1);
  vector<double> temp;
  temp.push_back(sorted_x(0));
  for (int i = 1; i < x.size(); i++) {
    //if(sorted_x(i) != *(temp.end()-1)) // Fucking important!!! "-1"!! or:
    if(sorted_x(i) != temp.back())
      temp.push_back(sorted_x(i));
  }
  return vec_assign_mat(temp, 1);
}

// matlab function: length
int length(MatrixXd x){
  return max(x.rows(), x.cols());
}

// matlab function: gradient
void gradient(MatrixXd x, MatrixXd& gx, MatrixXd& gy){
  int i, j;
  for (i = 0; i < x.rows(); i++) {
    gx(i, 0) = x(i, 1) - x(i, 0);
    gx(i, x.cols()-1) = x(i, x.cols()-1) - x(i, x.cols()-2);
  }
  for (i = 0; i < x.cols(); i++) {
    gy(0, i) = x(1, i) - x(0, i);
    gy(x.rows()-1, i) = x(x.rows()-1, i) - x(x.rows()-2, i);
  }
  for (i = 1; i < x.rows()-1; i++) {
    for (j = 1; j < x.cols()-1; j++) {
      gx(i, j) = (x(i, j+1) - x(i, j-1)) / 2;
      gy(i, j) = (x(i+1, j) - x(i-1, j)) / 2;
    }
  }
}


// matlab function: find
void mat_find(MatrixXd x, double obj, int *& pos_r, int *& pos_c, int& n){
  vector<double> x_ = mat_assign_vec(x);
  vector<int> pos;
  vector<double>::iterator it = x_.begin();
  while(it != x_.end()){
    if (*it == obj) {
      pos.push_back(it-x_.begin());
    }
    it++;
  }
  // assign coordinates
  // We need to manually release the spaces.
  pos_r = new int[pos.size()];
  pos_c = new int[pos.size()];
  for (int i = 0; i < pos.size(); i++) {
    pos_r[i] = pos[i] % x.rows();
    pos_c[i] = pos[i] / x.rows();
  }
  n = pos.size();
}

void blkproc(MatrixXd& x, int m, int n, MatrixXd fun(MatrixXd, int), int para1){
  int i, j;
  if (x.rows() % m != 0 || x.cols() % n != 0) {
    std::cout << "Warning! blkproc function can't slice image to perfect blocks" << std::endl;
    // Await to process this condition
    for (i = 0; i < x.rows()/m; i++) {
     for (j = 0; j < x.cols()/n; j++) {
       fun(x.block(i*(m-1), j*(n-1), i*m, j*n), para1); 
     }
    }
  }else{
    for (i = 0; i < x.rows()/m; i++) {
      for (j = 0; j < x.cols()/n; j++) {
        fun(x.block(i*(m-1), j*(n-1), i*m, j*n), para1); 
      }
    }
  }
}

cv::Mat mat_to_cvmat(MatrixXd obj){
  int nl = obj.rows();
  int nc = obj.cols();
  int i, j;
  cv::Mat _obj(nl, nc);
  for (i = 0; i < nl; i++) {
    uchar* data = _obj.ptr<uchar>(i);
    for (j = 0; j < nc; j++) {
      data[j] = obj(i, j);
    }
  }
  return _obj;
}

MatrixXd cvmat_to_mat(cv::Mat obj){
  int nl = obj.rows;
  int nc = obj.cols;
  int i, j;
  MatrixXd _obj(nl, nc);
  for (i = 0; i < nl; i++) {
    uchar* data = obj.ptr<uchar>(i);
    for (j = 0; j < nc; j++) {
      _obj(i, j) = data[j];
    }
  }
  return _obj;
}

//edge canny
MatrixXd edg_canny(MatrixXd src){
  MatrixXd dst;
  cv::Mat _src = mat_to_cvmat(src);
  cv::Canny(_src, _src, 1, 3, 3); // paras??/
  dst = cvmat_to_mat(_src);
  return dst;
}

// edge sobel
MatrixXd edge_sobel(MatrixXd src){
  MatrixXd dst;
  cv::Mat _src = mat_to_cvmat(src);
  cv::Sobel(_src, _src, CV_16S, 0, 1, 3);
  cv::Threshold(_src, _src, 2, 255, 3);
  dst = cvmat_to_mat(_src);
  return dst;
}

// 
