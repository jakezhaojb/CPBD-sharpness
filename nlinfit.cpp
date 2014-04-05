#include "nlinfit.h"

float model_fun(VectorXd beta, VectorXd x){
  float pred;
  assert(beta.size() == 4);
  //pred = (beta(0)-beta(1))/(1+exp(-(x-beta(2))/abs(beta(3))));

}

// Eigen不广播，建议重载操作符(+, -)
// x是一个vector，beta也是（所以不用改）
