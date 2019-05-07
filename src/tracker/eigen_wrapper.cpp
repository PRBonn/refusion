// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#include <tracker/eigen_wrapper.h>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/MatrixFunctions>

namespace refusion {

Eigen::Matrix<double, 6, 1> SolveLdlt(const Eigen::Matrix<double, 6, 6> &H,
                                      const Eigen::Matrix<double, 6, 1> &b) {
  return H.ldlt().solve(b);
}

Eigen::Matrix4d Exp(const Eigen::Matrix4d &mat) {
  return mat.exp();
}

}  // namespace refusion
