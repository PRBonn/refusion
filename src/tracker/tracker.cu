// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tracker.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "tracker/eigen_wrapper.h"
#include "utils/matrix_utils.h"
#include "utils/utils.h"
#include "utils/rgbd_image.h"

#define THREADS_PER_BLOCK3 32

namespace refusion {

Tracker::Tracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
                 const TrackerOptions &tracker_options,
                 const RgbdSensor &sensor) {
  cudaMallocManaged(&volume_, sizeof(tsdfvh::TsdfVolume));
  volume_->Init(tsdf_options);
  options_ = tracker_options;
  sensor_ = sensor;
  pose_ = Eigen::Matrix4d::Identity();
}

Tracker::~Tracker() {
  volume_->Free();
  cudaFree(volume_);
}

Eigen::Matrix4d v2t(const Vector6d &xi) {
  Eigen::Matrix4d M;

  M << 0.0  , -xi(2),  xi(1), xi(3),
       xi(2), 0.0   , -xi(0), xi(4),
      -xi(1), xi(0) , 0.0   , xi(5),
       0.0,   0.0   , 0.0   ,   0.0;

  return M;
}

__host__ __device__ float Intensity(float3 color) {
  return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
}

__host__ __device__ float ColorDifference(uchar3 c1, uchar3 c2) {
  float3 c1_float = ColorToFloat(c1);
  float3 c2_float = ColorToFloat(c2);
  return Intensity(c1_float)-Intensity(c2_float);
}

__global__ void CreateLinearSystem(tsdfvh::TsdfVolume *volume,
                                   float huber_constant, uchar3 *rgb,
                                   float *depth, bool *mask, float4x4 transform,
                                   RgbdSensor sensor, mat6x6 *acc_H,
                                   mat6x1 *acc_b, int downsample,
                                   float residuals_threshold,
                                   bool create_mask) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;
  for (int idx = index; idx < size / (downsample * downsample); idx += stride) {
    mat6x6 new_H;
    mat6x1 new_b;
    new_H.setZero();
    new_b.setZero();
    int v = (idx / (sensor.cols/downsample)) * downsample;
    int u = (idx - (sensor.cols/downsample) * v/downsample) * downsample;
    int i = v * sensor.cols + u;
    if (depth[i] < volume->GetOptions().min_sensor_depth) {
      continue;
    }
    if (depth[i] > volume->GetOptions().max_sensor_depth) {
      continue;
    }
    float3 point = transform * GetPoint3d(i, depth[i], sensor);
    tsdfvh::Voxel v1 = volume->GetInterpolatedVoxel(point);
    if (v1.weight == 0) {
      continue;
    }
    float sdf = v1.sdf;
    float3 color = make_float3(static_cast<float>(v1.color.x)/255,
                               static_cast<float>(v1.color.y)/255,
                               static_cast<float>(v1.color.z)/255);
    float3 color2 = make_float3(static_cast<float>(rgb[i].x)/255,
                                static_cast<float>(rgb[i].y)/255,
                                static_cast<float>(rgb[i].z)/255);
    if (sdf * sdf > residuals_threshold) {
      if (create_mask) mask[i] = true;
      continue;
    }
    mat1x3 gradient, gradient_color;
    // x
    float voxel_size = volume->GetOptions().voxel_size;
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(voxel_size, 0.0f, 0.0f));
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    tsdfvh::Voxel v2 = volume->GetInterpolatedVoxel(
        point + make_float3(-voxel_size, 0.0f, 0.0f));
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(0) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(0) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
    // y
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, voxel_size, 0.0f));
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    v2 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, -voxel_size, 0.0f));
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(1) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(1) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
    // z
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, 0.0f, voxel_size));
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    v2 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, 0.0f, -voxel_size));
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(2) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(2) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);

    // Partial derivative of position wrt optimization parameters
    mat3x6 d_position;
    d_position(0, 0) = 0;
    d_position(0, 1) = point.z;
    d_position(0, 2) = -point.y;
    d_position(0, 3) = 1;
    d_position(0, 4) = 0;
    d_position(0, 5) = 0;
    d_position(1, 0) = -point.z;
    d_position(1, 1) = 0;
    d_position(1, 2) = point.x;
    d_position(1, 3) = 0;
    d_position(1, 4) = 1;
    d_position(1, 5) = 0;
    d_position(2, 0) = point.y;
    d_position(2, 1) = -point.x;
    d_position(2, 2) = 0;
    d_position(2, 3) = 0;
    d_position(2, 4) = 0;
    d_position(2, 5) = 1;

    // Jacobian
    mat1x6 jacobian = gradient * d_position;
    mat1x6 jacobian_color = gradient_color * d_position;

    float huber = fabs(sdf) < huber_constant ? 1.0 : huber_constant/fabs(sdf);
    bool use_depth = true;
    bool use_color = true;
    float weight = 0.025;
    if (use_depth) {
      new_H = new_H + huber * jacobian.getTranspose() * jacobian;
      new_b = new_b + huber * jacobian.getTranspose() * sdf;
    }

    if (use_color) {
      new_H = new_H + weight * jacobian_color.getTranspose() * jacobian_color;
      new_b = new_b +
              weight * jacobian_color.getTranspose() *
                  (Intensity(color) - Intensity(color2));
    }

    for (int j = 0; j < 36; j++) atomicAdd(&((*acc_H)(j)), new_H(j));
    for (int j = 0; j < 6; j++) atomicAdd(&((*acc_b)(j)), new_b(j));
  }
}

void Tracker::TrackCamera(const RgbdImage &image, bool *mask,
                          bool create_mask) {
  Vector6d increment, prev_increment;
  increment << 0, 0, 0, 0, 0, 0;
  prev_increment = increment;

  mat6x6 *acc_H;
  cudaMallocManaged(&acc_H, sizeof(mat6x6));
  mat6x1 *acc_b;
  cudaMallocManaged(&acc_b, sizeof(mat6x1));
  cudaDeviceSynchronize();
  for (int lvl = 0; lvl < 3; ++lvl) {
    for (int i = 0; i < options_.max_iterations_per_level[lvl]; ++i) {
      Eigen::Matrix4d cam_to_world = Exp(v2t(increment)) * pose_;
      Eigen::Matrix4f cam_to_worldf = cam_to_world.cast<float>();
      float4x4 transform_cuda = float4x4(cam_to_worldf.data()).getTranspose();

      acc_H->setZero();
      acc_b->setZero();
      int threads_per_block = THREADS_PER_BLOCK3;
      int thread_blocks =
          (sensor_.cols * sensor_.rows + threads_per_block - 1) /
          threads_per_block;
      bool create_mask_now =
          (lvl == 2) && (i == (options_.max_iterations_per_level[2] - 1)) &&
          create_mask;

      float residuals_threshold = 0;
      residuals_threshold = volume_->GetOptions().truncation_distance *
                            volume_->GetOptions().truncation_distance / 2;
      if (!create_mask) {
        residuals_threshold = volume_->GetOptions().truncation_distance *
                              volume_->GetOptions().truncation_distance;
      }
      // Kernel to fill in parallel acc_H and acc_b
      CreateLinearSystem<<<thread_blocks, threads_per_block>>>(
          volume_, options_.huber_constant, image.rgb_, image.depth_, mask,
          transform_cuda, sensor_, acc_H, acc_b, options_.downsample[lvl],
          residuals_threshold, create_mask_now);
      cudaDeviceSynchronize();
      Eigen::Matrix<double, 6, 6> H;
      Vector6d b;
      for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
          H(r, c) = static_cast<double>((*acc_H)(r, c));
        }
      }
      for (int k = 0; k < 6; k++) {
        b(k) = static_cast<double>((*acc_b)(k));
      }
      double scaling = 1 / H.maxCoeff();
      b *= scaling;
      H *= scaling;
      H = H + options_.regularization * Eigen::MatrixXd::Identity(6, 6) * i;
      increment = increment - SolveLdlt(H, b);
      Vector6d change = increment - prev_increment;
      if (change.norm() <= options_.min_increment) break;
      prev_increment = increment;
    }
  }
  if (std::isnan(increment.sum())) increment << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  cudaFree(acc_H);
  cudaFree(acc_b);

  pose_ = Exp(v2t(increment)) * pose_;
  prev_increment_ = increment;
}

void ApplyMaskFlood(const cv::Mat &depth, cv::Mat &mask, float threshold) {
  int erosion_size = 15;
  cv::Mat erosion_kernel = cv::getStructuringElement(
  cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
  cv::Point(erosion_size, erosion_size));
  cv::Mat eroded_mask;
  cv::erode(mask, eroded_mask, erosion_kernel);
  std::vector<std::pair<int, int>> mask_vector;
  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      mask.at<uchar>(i, j) = 0;
      if (eroded_mask.at<uchar>(i, j) > 0) {
        mask_vector.push_back(std::make_pair(i, j));
      }
    }
  }

  while (!mask_vector.empty()) {
    int i = mask_vector.back().first;
    int j = mask_vector.back().second;
    mask_vector.pop_back();
    if (depth.at<float>(i, j) > 0 && mask.at<uchar>(i, j) == 0) {
      float old_depth = depth.at<float>(i, j);
      mask.at<uchar>(i, j) = 255;
      if (i - 1 >= 0) {  // up
        if (depth.at<float>(i - 1, j) > 0 && mask.at<uchar>(i-1, j) == 0 &&
            fabs(depth.at<float>(i - 1, j) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i - 1, j));
        }
      }
      if (i + 1 < depth.rows) {  // down
        if (depth.at<float>(i + 1, j) > 0 && mask.at<uchar>(i+1, j) == 0 &&
            fabs(depth.at<float>(i + 1, j) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i + 1, j));
        }
      }
      if (j - 1 >= 0) {  // left
        if (depth.at<float>(i, j - 1) > 0 && mask.at<uchar>(i, j-1) == 0 &&
            fabs(depth.at<float>(i, j - 1) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i, j - 1));
        }
      }
      if (j + 1 < depth.cols) {  // right
        if (depth.at<float>(i, j + 1) > 0 && mask.at<uchar>(i, j+1) == 0 &&
            fabs(depth.at<float>(i, j + 1) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i, j + 1));
        }
      }
    }
  }
}

void Tracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth) {
  RgbdImage image;
  image.Init(sensor_);

  // Linear copy for now
  for (int i = 0; i < image.sensor_.rows; i++) {
    for (int j = 0; j < image.sensor_.cols; j++) {
      image.rgb_[i * image.sensor_.cols + j] =
          make_uchar3(rgb.at<cv::Vec3b>(i, j)(2), rgb.at<cv::Vec3b>(i, j)(1),
                      rgb.at<cv::Vec3b>(i, j)(0));
      image.depth_[i * image.sensor_.cols + j] = depth.at<float>(i, j);
    }
  }

  bool *mask;
  cudaMallocManaged(&mask,
                    sizeof(bool) * image.sensor_.rows * image.sensor_.cols);
  for (int i = 0; i < image.sensor_.rows * image.sensor_.cols; i++) {
    mask[i] = false;
  }

  if (!first_scan_) {
    Eigen::Matrix4d prev_pose = pose_;
    TrackCamera(image, mask, true);

    cv::Mat cvmask(image.sensor_.rows, image.sensor_.cols, CV_8UC1);
    for (int i = 0; i < image.sensor_.rows; i++) {
      for (int j = 0; j < image.sensor_.cols; j++) {
        if (mask[i * image.sensor_.cols + j]) {
          cvmask.at<uchar>(i, j) = 255;
        } else {
          cvmask.at<uchar>(i, j) = 0;
        }
      }
    }

    ApplyMaskFlood(depth,cvmask,0.007);

    int dilation_size = 10;
    cv::Mat dilation_kernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
    cv::Point(dilation_size, dilation_size));
    cv::dilate(cvmask, cvmask, dilation_kernel);

    for (int i = 0; i < image.sensor_.rows; i++) {
      for (int j = 0; j < image.sensor_.cols; j++) {
        if (cvmask.at<uchar>(i, j) > 0) {
          mask[i * image.sensor_.cols + j] = true;
        } else {
          mask[i * image.sensor_.cols + j] = false;
        }
      }
    }

    pose_ = prev_pose;
    TrackCamera(image, mask, false);
  } else {
    first_scan_ = false;
  }
  Eigen::Matrix4f posef = pose_.cast<float>();
  float4x4 pose_cuda = float4x4(posef.data()).getTranspose();

  volume_->IntegrateScan(image, pose_cuda, mask);

  cudaFree(mask);
}

Eigen::Matrix4d Tracker::GetCurrentPose() {
  return pose_;
}

tsdfvh::Mesh Tracker::ExtractMesh(const float3 &lower_corner,
                                  const float3 &upper_corner) {
  return volume_->ExtractMesh(lower_corner, upper_corner);
}

cv::Mat Tracker::GenerateRgb(int width, int height) {
  Eigen::Matrix4f posef = pose_.cast<float>();
  float4x4 pose_cuda = float4x4(posef.data()).getTranspose();
  RgbdSensor virtual_sensor;
  virtual_sensor.rows = height;
  virtual_sensor.cols = width;
  virtual_sensor.depth_factor = sensor_.depth_factor;
  float factor_x = static_cast<float>(virtual_sensor.cols) /
                   static_cast<float>(sensor_.cols);
  float factor_y = static_cast<float>(virtual_sensor.rows) /
                   static_cast<float>(sensor_.rows);
  virtual_sensor.fx = factor_x * sensor_.fx;
  virtual_sensor.fy = factor_y * sensor_.fy;
  virtual_sensor.cx = factor_x * sensor_.cx;
  virtual_sensor.cy = factor_y * sensor_.cy;
  uchar3 *virtual_rgb = volume_->GenerateRgb(pose_cuda, virtual_sensor);

  cv::Mat cv_virtual_rgb(virtual_sensor.rows, virtual_sensor.cols, CV_8UC3);
  for (int i = 0; i < virtual_sensor.rows; i++) {
    for (int j = 0; j < virtual_sensor.cols; j++) {
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[2] =
          virtual_rgb[i * virtual_sensor.cols + j].x;
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[1] =
          virtual_rgb[i * virtual_sensor.cols + j].y;
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[0] =
          virtual_rgb[i * virtual_sensor.cols + j].z;
    }
  }

  return cv_virtual_rgb;
}

}  // namespace refusion
