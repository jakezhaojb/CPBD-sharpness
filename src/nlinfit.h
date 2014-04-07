#ifndef _NLINFIT_H_
#define _NLINFIT_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

ArrayXd model_fun(Vector4d beta, VectorXd x);
double compute_cost(Vector4d beta, VectorXd x, VectorXd y);

template<typename T>
T sgn(T num){
  return (num>0)?num:-num;
}

Vector4d compute_grad(Vector4d beta, VectorXd x, VectorXd y);
VectorXd batch_grad_desc(Vector4d beta_init, VectorXd x, VectorXd y, double step_size=0.01, int MaxIter=2000, double Facttol=1e-5);

#endif
