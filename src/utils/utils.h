// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "utils/rgbd_sensor.h"

namespace refusion {

__host__ __device__ inline float norm(const float3 &vec) {
  return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__host__ __device__ inline float3 normalize(const float3 &vec) {
  float vec_norm = norm(vec);
  return make_float3(vec.x / vec_norm, vec.y / vec_norm, vec.z / vec_norm);
}

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline int3 operator+(const int3 &a, const int3 &b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float b, const float3 &a) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator/(const float3 &a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline float distance(const float3 &a, const float3 &b) {
  return norm(b - a);
}

__host__ __device__ inline int sign(float n) { return (n > 0) - (n < 0); }

__host__ __device__ inline float signf(float value) {
  return (value > 0) - (value < 0);
}

__host__ __device__ inline float3 ColorToFloat(uchar3 c) {
  return make_float3(static_cast<float>(c.x)/255,
                            static_cast<float>(c.y)/255,
                            static_cast<float>(c.z)/255);
}

__host__ __device__ inline float3 GetPoint3d(int i, float depth, RgbdSensor sensor) {
  int v = i / sensor.cols;
  int u = i - sensor.cols * v;
  float3 point;
  point.z = depth;
  point.x = (static_cast<float>(u) - sensor.cx) * point.z / sensor.fx;
  point.y = (static_cast<float>(v) - sensor.cy) * point.z / sensor.fy;
  return point;
}

__host__ __device__ inline int2 Project(float3 point3d, RgbdSensor sensor) {
  float2 point2df;
  point2df.x = (sensor.fx * point3d.x) / point3d.z + sensor.cx;
  point2df.y = (sensor.fy * point3d.y) / point3d.z + sensor.cy;
  return make_int2(round(point2df.x), round(point2df.y));
}

}  // namespace refusion
