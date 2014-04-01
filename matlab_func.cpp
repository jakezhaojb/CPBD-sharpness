#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <string>
#include <Eigen/Dense>
//#include <cv.h>
//#include <highgui.h>

using namespace std;
//using namespace cv;
using namespace Eigen;

// Transform from matlab matrix to Eigen::MatrixXd
vector<double> mat_assign_vec(MatrixXd x){
  vector<double> obj;
  for (int i = 0; i < x.size(); i++) {
    obj.push_back(x(i));
  }
  return obj;
}

VectorXd vec_assign_mat(vector<double> x, bool order){
  if (!order) {
    // Fortran order, ColVector
    VectorXd obj(x.size());
    for (int i = 0; i < x.size(); i++) {
      obj(i) = x[i];
    }
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

MatrixXd vec_assign_mat(vector<double> x, int rows, int cols, bool order){
  int i, j;
  MatrixXd obj(rows, cols);
  if (order) {
   r // C++ order, row
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


