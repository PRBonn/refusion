// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#pragma once
#include <Eigen/Core>

namespace refusion {

Eigen::Matrix<double, 6, 1> SolveLdlt(const Eigen::Matrix<double, 6, 6> &H,
                                      const Eigen::Matrix<double, 6, 1> &b);
Eigen::Matrix4d Exp(const Eigen::Matrix4d &mat);

Eigen::Matrix4d Inverse(const Eigen::Matrix4d &mat);

}  // namespace refusion
