// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#include "tsdfvh/tsdf_volume.h"
#include <cfloat>
#include <cmath>
#include "marching_cubes/mesh_extractor.h"

#define THREADS_PER_BLOCK2 64

namespace refusion {

namespace tsdfvh {

void TsdfVolume::Init(const TsdfVolumeOptions &options) {
  options_ = options;
  HashTable::Init(options_.num_buckets, options_.bucket_size,
                  options_.num_blocks, options_.block_size);
}

void TsdfVolume::Free() { HashTable::Free(); }

__host__ __device__ float3 TsdfVolume::GlobalVoxelToWorld(int3 position) {
  return make_float3(position.x * options_.voxel_size,
                     position.y * options_.voxel_size,
                     position.z * options_.voxel_size);
}

__host__ __device__ int3 TsdfVolume::WorldToGlobalVoxel(float3 position) {
  return make_int3(position.x / options_.voxel_size + signf(position.x) * 0.5f,
                   position.y / options_.voxel_size + signf(position.y) * 0.5f,
                   position.z / options_.voxel_size + signf(position.z) * 0.5f);
}

__host__ __device__ int3 TsdfVolume::WorldToBlock(float3 position) {
  int3 voxel_position = WorldToGlobalVoxel(position);
  int3 block_position;
  if (voxel_position.x < 0)
    block_position.x = (voxel_position.x - block_size_ + 1) / block_size_;
  else
    block_position.x = voxel_position.x / block_size_;

  if (voxel_position.y < 0)
    block_position.y = (voxel_position.y - block_size_ + 1) / block_size_;
  else
    block_position.y = voxel_position.y / block_size_;

  if (voxel_position.z < 0)
    block_position.z = (voxel_position.z - block_size_ + 1) / block_size_;
  else
    block_position.z = voxel_position.z / block_size_;

  return block_position;
}

__host__ __device__ int3 TsdfVolume::WorldToLocalVoxel(float3 position) {
  int3 position_global = WorldToGlobalVoxel(position);
  int3 position_local = make_int3(position_global.x % block_size_,
                                  position_global.y % block_size_,
                                  position_global.z % block_size_);
  if (position_local.x < 0) position_local.x += block_size_;
  if (position_local.y < 0) position_local.y += block_size_;
  if (position_local.z < 0) position_local.z += block_size_;
  return position_local;
}

__host__ __device__ Voxel TsdfVolume::GetVoxel(float3 position) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);
  HashEntry entry = HashTable::FindHashEntry(block_position);
  if (entry.pointer == kFreeEntry) {
    Voxel voxel;
    voxel.sdf = 0;
    voxel.color = make_uchar3(0, 0, 0);
    voxel.weight = 0;
    return voxel;
  }
  return HashTable::voxel_blocks_[entry.pointer].at(local_voxel);
}

__host__ __device__ Voxel TsdfVolume::GetInterpolatedVoxel(float3 position) {
  Voxel v0 = GetVoxel(position);
  if (v0.weight == 0) return v0;
  float voxel_size = options_.voxel_size;
  const float3 pos_dual =
      position -
      make_float3(voxel_size / 2.0f, voxel_size / 2.0f, voxel_size / 2.0f);
  float3 voxel_position = position / voxel_size;
  float3 weight = make_float3(voxel_position.x - floor(voxel_position.x),
                              voxel_position.y - floor(voxel_position.y),
                              voxel_position.z - floor(voxel_position.z));

  float distance = 0.0f;
  float3 color_float = make_float3(0.0f, 0.0f, 0.0f);
  float3 vColor;

  Voxel v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, 0.0f));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance +=
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float +
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance +=
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float +
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, 0.0f));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, 0.0f));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, voxel_size));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, 0.0f));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * weight.y * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, voxel_size));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * weight.y * weight.z * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, voxel_size));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * (1.0f - weight.y) * weight.z * v0.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, voxel_size));
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * weight.y * weight.z * v0.sdf;
    color_float = color_float + weight.x * weight.y * weight.z * vColor;

  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * weight.y * weight.z * v.sdf;
    color_float = color_float + weight.x * weight.y * weight.z * vColor;
  }

  uchar3 color = make_uchar3(color_float.x, color_float.y, color_float.z);
  v.weight = v0.weight;
  v.sdf = distance;
  v.color = color;
  return v;
}

__host__ __device__ bool TsdfVolume::SetVoxel(float3 position,
                                              const Voxel &voxel) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);
  HashEntry entry = HashTable::FindHashEntry(block_position);
  if (entry.pointer == kFreeEntry) {
    return false;
  }
  HashTable::voxel_blocks_[entry.pointer].at(local_voxel) = voxel;
  return true;
}

__host__ __device__ bool TsdfVolume::UpdateVoxel(float3 position,
                                                 const Voxel &voxel) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);
  HashEntry entry = HashTable::FindHashEntry(block_position);
  if (entry.pointer == kFreeEntry) {
    return false;
  }
  HashTable::voxel_blocks_[entry.pointer]
      .at(local_voxel)
      .Combine(voxel, options_.max_sdf_weight);
  return true;
}

__global__ void AllocateFromDepthKernel(TsdfVolume *volume, float *depth,
                                        RgbdSensor sensor, float4x4 transform) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float truncation_distance = volume->GetOptions().truncation_distance;
  float block_size =
      volume->GetOptions().block_size * volume->GetOptions().voxel_size;

  float3 start_pt = make_float3(transform.m14, transform.m24, transform.m34);
  for (int i = index; i < size; i += stride) {
    if (depth[i] < volume->GetOptions().min_sensor_depth ||
        depth[i] > volume->GetOptions().max_sensor_depth)
      continue;
    float3 point = GetPoint3d(i, depth[i], sensor);
    point = transform * point;
    if (point.x == 0 && point.y == 0 && point.z == 0) continue;
    // compute start and end of the ray
    float3 ray_direction = normalize(point - start_pt);
    float surface_distance = distance(start_pt, point);
    float3 ray_start = start_pt;
    float3 ray_end =
        start_pt + ray_direction * (surface_distance + truncation_distance);
    // traverse the ray discretely using the block size and allocate it
    // adapted from https://github.com/francisengelmann/fast_voxel_traversal/blob/master/main.cpp
    int3 block_start = make_int3(floor(ray_start.x / block_size),
                                 floor(ray_start.y / block_size),
                                 floor(ray_start.z / block_size));

    int3 block_end = make_int3(floor(ray_end.x / block_size),
                               floor(ray_end.y / block_size),
                               floor(ray_end.z / block_size));

    int3 block_position = block_start;
    int3 step = make_int3(sign(ray_direction.x),
                          sign(ray_direction.y),
                          sign(ray_direction.z));

    float3 delta_t;
    delta_t.x =
        (ray_direction.x != 0) ? fabs(block_size / ray_direction.x) : FLT_MAX;
    delta_t.y =
        (ray_direction.y != 0) ? fabs(block_size / ray_direction.y) : FLT_MAX;
    delta_t.z =
        (ray_direction.z != 0) ? fabs(block_size / ray_direction.z) : FLT_MAX;

    float3 boundary = make_float3(
        (block_position.x + static_cast<float>(step.x)) * block_size,
        (block_position.y + static_cast<float>(step.y)) * block_size,
        (block_position.z + static_cast<float>(step.z)) * block_size);

    float3 max_t;
    max_t.x = (ray_direction.x != 0)
                  ? (boundary.x - ray_start.x) / ray_direction.x
                  : FLT_MAX;
    max_t.y = (ray_direction.y != 0)
                  ? (boundary.y - ray_start.y) / ray_direction.y
                  : FLT_MAX;
    max_t.z = (ray_direction.z != 0)
                  ? (boundary.z - ray_start.z) / ray_direction.z
                  : FLT_MAX;

    int3 diff = make_int3(0, 0, 0);
    bool neg_ray = false;

    if (block_position.x != block_end.x && ray_direction.x < 0) {
      diff.x--;
      neg_ray = true;
    }
    if (block_position.y != block_end.y && ray_direction.y < 0) {
      diff.y--;
      neg_ray = true;
    }
    if (block_position.z != block_end.z && ray_direction.z < 0) {
      diff.z--;
      neg_ray = true;
    }
    volume->AllocateBlock(block_position);

    if (neg_ray) {
      block_position = block_position + diff;
      volume->AllocateBlock(block_position);
    }

    while (block_position.x != block_end.x || block_position.y != block_end.y ||
           block_position.z != block_end.z) {
      if (max_t.x < max_t.y) {
        if (max_t.x < max_t.z) {
          block_position.x += step.x;
          max_t.x += delta_t.x;
        } else {
          block_position.z += step.z;
          max_t.z += delta_t.z;
        }
      } else {
        if (max_t.y < max_t.z) {
          block_position.y += step.y;
          max_t.y += delta_t.y;
        } else {
          block_position.z += step.z;
          max_t.z += delta_t.z;
        }
      }
      volume->AllocateBlock(block_position);
    }
  }
}

__global__ void IntegrateScanKernel(TsdfVolume *volume, uchar3 *color,
                                    float *depth, RgbdSensor sensor,
                                    float4x4 transform, float4x4 inv_transform,
                                    bool *mask) {
  //loop through ALL entries
  //  if entry is in camera frustum
  //    loop through voxels inside block
    //    if voxel is in truncation region
    //      update voxels
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int block_size = volume->GetOptions().block_size;
  float voxel_size = volume->GetOptions().voxel_size;
  float truncation_distance = volume->GetOptions().truncation_distance;

  for (int i = index; i < volume->GetNumEntries(); i += stride) {
    float3 position = make_float3(
        volume->GetHashEntry(i).position.x * voxel_size * block_size,
        volume->GetHashEntry(i).position.y * voxel_size * block_size,
        volume->GetHashEntry(i).position.z * voxel_size * block_size);
    // To camera coordinates
    float3 position_cam = inv_transform * position;
    // If behind camera plane discard
    if (position_cam.z < 0) continue;
    float3 block_center =
        make_float3(position_cam.x + 0.5 * voxel_size * block_size,
                    position_cam.y + 0.5 * voxel_size * block_size,
                    position_cam.z + 0.5 * voxel_size * block_size);
    int2 image_position = Project(block_center, sensor);
    if (image_position.x >= 0 && image_position.y >= 0 &&
        image_position.x < sensor.cols && image_position.y < sensor.rows) {
      float3 start_pt = make_float3(0, 0, 0);

      for (int bx = 0; bx < block_size; bx++) {
        for (int by = 0; by < block_size; by++) {
          for (int bz = 0; bz < block_size; bz++) {
            float3 voxel_position = make_float3(position.x + bx * voxel_size,
                                                position.y + by * voxel_size,
                                                position.z + bz * voxel_size);
            voxel_position = inv_transform * voxel_position;
            image_position = Project(voxel_position, sensor);
            // Check again inside the block
            if (image_position.x >= 0 && image_position.y >= 0 &&
                image_position.x < sensor.cols &&
                image_position.y < sensor.rows) {
              int idx = image_position.y * sensor.cols + image_position.x;
              if (mask[idx]) continue;
              if (depth[idx] < volume->GetOptions().min_sensor_depth) continue;
              if (depth[idx] > volume->GetOptions().max_sensor_depth) continue;
              float3 point3d = GetPoint3d(idx, depth[idx], sensor);
              float surface_distance = distance(start_pt, point3d);
              float voxel_distance = distance(start_pt, voxel_position);
              if (voxel_distance > surface_distance - truncation_distance &&
                  voxel_distance < surface_distance + truncation_distance &&
                  (depth[idx] < volume->GetOptions().max_sensor_depth)) {
                Voxel voxel;
                voxel.sdf = surface_distance - voxel_distance;
                voxel.color = color[idx];
                voxel.weight = (unsigned char)1;
                // To world coordinates
                voxel_position = transform * voxel_position;
                volume->UpdateVoxel(voxel_position, voxel);
              } else if (voxel_distance <
                         surface_distance - truncation_distance) {
                voxel_position = transform * voxel_position;
                Voxel voxel;
                voxel.sdf = truncation_distance;
                voxel.color = color[idx];
                voxel.weight = (unsigned char)1;
                volume->UpdateVoxel(voxel_position, voxel);
              }
            }
          }
        }
      }  // End single block update
    }
  }
}

void TsdfVolume::IntegrateScan(const RgbdImage &image, float4x4 camera_pose,
                               bool *mask) {
  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (options_.num_buckets * options_.bucket_size + threads_per_block - 1) /
      threads_per_block;

  AllocateFromDepthKernel<<<thread_blocks, threads_per_block>>>(
      this, image.depth_, image.sensor_, camera_pose);
  cudaDeviceSynchronize();
  float4x4 inv_camera_pose = camera_pose.getInverse();
  IntegrateScanKernel<<<thread_blocks, threads_per_block>>>(
      this, image.rgb_, image.depth_, image.sensor_, camera_pose,
      inv_camera_pose, mask);
  cudaDeviceSynchronize();
}

__global__ void GenerateDepthKernel(TsdfVolume *volume, RgbdSensor sensor,
                                    float4x4 camera_pose,
                                    float *virtual_depth) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float3 start_pt =
      make_float3(camera_pose.m14, camera_pose.m24, camera_pose.m34);
  for (int i = index; i < size; i += stride) {
    float current_depth = 0;
    while (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point);
      if (v.weight == 0) {
        current_depth += volume->GetOptions().truncation_distance;
      } else {
        current_depth += v.sdf;
      }
      if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size) break;
    }
    virtual_depth[i] = current_depth;
  }
}

__global__ void GenerateRgbKernel(TsdfVolume *volume, RgbdSensor sensor,
                                  float4x4 camera_pose, uchar3 *virtual_rgb) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float3 start_pt =
      make_float3(camera_pose.m14, camera_pose.m24, camera_pose.m34);
  for (int i = index; i < size; i += stride) {
    float current_depth = 0;
    while (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point);
      if (v.weight == 0) {
        current_depth += volume->GetOptions().truncation_distance;
      } else {
        current_depth += v.sdf;
      }
      if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size) break;
    }
    if (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point);
      virtual_rgb[i] = v.color;
    } else {
      virtual_rgb[i] = make_uchar3(0, 0, 0);
    }
  }
}

float* TsdfVolume::GenerateDepth(float4x4 camera_pose, RgbdSensor sensor) {
  float* virtual_depth;
  cudaMallocManaged(&virtual_depth, sizeof(float) * sensor.rows * sensor.cols);
  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (sensor.rows * sensor.cols + threads_per_block - 1) / threads_per_block;
  GenerateDepthKernel<<<thread_blocks, threads_per_block>>>(
      this, sensor, camera_pose, virtual_depth);
  cudaDeviceSynchronize();
  return virtual_depth;
}

uchar3* TsdfVolume::GenerateRgb(float4x4 camera_pose, RgbdSensor sensor) {
  uchar3* virtual_rgb;
  cudaMallocManaged(&virtual_rgb, sizeof(float) * sensor.rows * sensor.cols);
  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (sensor.rows * sensor.cols + threads_per_block - 1) / threads_per_block;
  GenerateRgbKernel<<<thread_blocks, threads_per_block>>>(
      this, sensor, camera_pose, virtual_rgb);
  cudaDeviceSynchronize();
  return virtual_rgb;
}

Mesh TsdfVolume::ExtractMesh(const float3 &lower_corner,
                             const float3 &upper_corner) {
  MeshExtractor *mesh_extractor;
  cudaMallocManaged(&mesh_extractor, sizeof(MeshExtractor));
  mesh_extractor->Init(2000000, options_.voxel_size);
  mesh_extractor->ExtractMesh(this, lower_corner, upper_corner);
  Mesh *mesh;
  cudaMallocManaged(&mesh, sizeof(Mesh));
  *mesh = mesh_extractor->GetMesh();
  return *mesh;
}

__host__ __device__ TsdfVolumeOptions TsdfVolume::GetOptions() {
  return options_;
}

}  // namespace tsdfvh

}  // namespace refusion
