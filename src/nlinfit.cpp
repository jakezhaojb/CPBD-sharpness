#include "nlinfit.h"

#define debug 1

// x & y, feature dimension: 1-D
// beta, parameters to be fitted
ArrayXd model_fun(Vector4d beta, VectorXd x){
  ArrayXd pred;
  pred = (beta(0)-beta(1))/(1+exp(-(x.array()-beta(2))/abs(beta(3))));

  return pred;
}

double compute_cost(Vector4d beta, VectorXd x, VectorXd y){
  double cost;
  assert(x.size()==y.size());
  ArrayXd pred, cost_array;
  pred = model_fun(beta, x);
  cost_array = (pred - y.array()).pow(2);
  cost = cost_array.sum() / (2.*x.size());

  return cost;
}

Vector4d compute_grad(Vector4d beta, VectorXd x, VectorXd y){
  Vector4d grad;
  ArrayXd tmp;
  ArrayXd pred = model_fun(beta, x);

  assert(x.size()==y.size());

  // beta(0)
  tmp = 1 / (1 + exp(-(x.array()-beta(2))/abs(beta(3))));
  tmp *= pred - y.array();
  grad(0) = tmp.sum() / x.size();

  // beta(1)
  tmp = 1 / (1 + exp(-(x.array()-beta(2))/abs(beta(3))));
  tmp = 1 - tmp;
  tmp *= pred - y.array();
  grad(1) = tmp.sum() / x.size();

  // beta(2)
  tmp = -(beta(0)- beta(1)) * (exp((beta(2)-x.array())/abs(beta(3)))/abs(beta(3))) \
        / (1+exp((beta(2)-x.array())/abs(beta(3)))).pow(2);
  tmp *= pred - y.array();
  grad(2) = tmp.sum() / x.size();

  // beta(3)
  tmp = (beta(0) - beta(1)) * (beta(2)-x.array()).pow(2) * sgn(beta(3)) \
        / (abs(beta(3))*pow(beta(3), 2)*(1+exp((beta(2)-x.array())/abs(beta(3)))).pow(2));
  tmp *= pred - y.array();
  grad(3) = tmp.sum() / x.size();

  return grad;
}

VectorXd batch_grad_desc(Vector4d beta_init, VectorXd x, VectorXd y, double step_size, int MaxIter, double Facttol){
  int i;
  Vector4d beta, beta_grad;
  double cost, cost_;

  beta = beta_init;
  if (debug) {
    std::cout << beta << std::endl;
    cin.get();
  }
  cost = compute_cost(beta, x, y);
  for (i = 0; i < MaxIter; i++) {
    cost_ = cost;
    beta_grad = compute_grad(beta, x, y);
    beta -= (beta_grad.array() * step_size).matrix();
    cost = compute_cost(beta, x, y);
    std::cout << "The cost is: " << cost << std::endl;
    if (abs(cost-cost_)/max(max(1.0, cost_), cost) < Facttol) {
      if (debug) {
      }
      std::cout << "Gradient Descent Converges!" << std::endl;
      break;
    }
  }
  return beta;
}

int main(int argc, const char *argv[])
{
  VectorXd x = VectorXd::Random(8002);
  VectorXd y = VectorXd::Random(x.size());
  Vector4d beta = VectorXd::Random(4);
  //beta << 0.9, 0.1, 0.5, 1.0;
  if (debug) {
    std::cout << beta << std::endl;
    cin.get();
    //std::cout << x.transpose() << std::endl;
    //std::cout << "--------------------------" << std::endl;
    //std::cout << y.transpose() << std::endl;
    //std::cout << "--------------------------" << std::endl;
  }
  std::cout << batch_grad_desc(beta, x, y) << std::endl;

  return 0;
}
