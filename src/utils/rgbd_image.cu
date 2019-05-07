// Copyright 2018 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#include "rgbd_image.h"

namespace refusion {

RgbdImage::~RgbdImage() {
  cudaDeviceSynchronize();
  cudaFree(rgb_);
  cudaFree(depth_);
}

void RgbdImage::Init(const RgbdSensor &sensor) {
  sensor_ = sensor;
  cudaMallocManaged(&rgb_, sizeof(uchar3) * sensor_.rows * sensor.cols);
  cudaMallocManaged(&depth_, sizeof(float) * sensor_.rows * sensor.cols);
  cudaDeviceSynchronize();
}

__host__ __device__ inline float3 RgbdImage::GetPoint3d(int u, int v) const {
  float3 point;
  point.z = depth_[v * sensor_.cols + u];
  point.x = (static_cast<float>(u) - sensor_.cx) * point.z / sensor_.fx;
  point.y = (static_cast<float>(v) - sensor_.cy) * point.z / sensor_.fy;
  return point;
}

__host__ __device__ inline float3 RgbdImage::GetPoint3d(int i) const {
  int v = i / sensor_.cols;
  int u = i - sensor_.rows * v;
  return GetPoint3d(u, v);
}

}  // namespace refusion
