#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::H_function(const Eigen::VectorXd &x_)
{

  VectorXd p(3);
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / (sqrt(px * px + py * py));

  p << rho, phi, rho_dot;
  return p;
}

VectorXd KalmanFilter::inverse_H_function(const Eigen::VectorXd &x_)
{
  VectorXd c(4);
  double rho = x_(0);
  double phi = x_(1);
  double rho_dot = x_(2);

  double px = rho * cos(phi);
  double py = rho * sin(phi);
  if (px < 0.0001)
    px = 0.00001;
  if (py < 0.00001)
    py = 0.00001;

  double vx = rho_dot * cos(phi);
  double vy = rho_dot * sin(phi);

  c << px, py, vx, vy;

  return c;
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  /**
    * update the state by using Extended Kalman Filter equations
  */

  VectorXd z_pred = H_function(x_);

  VectorXd y = z - z_pred;

  while (y(1) > M_PI || y(1) < -1 * M_PI)
  {
    if (y(1) > M_PI)
      y(1) -= M_PI;
    else
      y(1) += M_PI;
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  // x_ = inverse_H_function(x_);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
