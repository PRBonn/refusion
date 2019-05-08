// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once
#include "tsdfvh/hash_table.h"
#include "utils/rgbd_image.h"
#include "marching_cubes/mesh.h"
#include "utils/matrix_utils.h"
#include "utils/utils.h"

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Options for the TSDF representation (see TsdfVolume).
 */
struct TsdfVolumeOptions {
    /** Size of the side of a voxel (in meters) */
    float voxel_size;

    /** Total number of buckets in the table */
    int num_buckets;

    /** Maximum number of entries in a bucket */
    int bucket_size;

    /** Maximum number of blocks that can be allocated */
    int num_blocks;

    /** Size in voxels of the side of a voxel block */
    int block_size;

    /** Maximum weight that a voxel can have */
    int max_sdf_weight;

    /** Truncation distance of the TSDF */
    float truncation_distance;

    /**
     * Maximum sensor depth that is considered. Higher depth values are 
     * discarded 
     */
    float max_sensor_depth;

    /**
     * Minimum sensor depth that is considered. Lower depth values are discarded
     */
    float min_sensor_depth;
};

/**
 * @brief      Class that represents a TSDF volume. It handles access and
 *             manipulation of voxels in world coordinates, scan integration
 *             from an RGB-D sensor, and mesh extraction.
 */
class TsdfVolume : public HashTable {
 public:
  /**
   * @brief      Initializes the TSDF volume.
   *
   * @param[in]  options  The options for the TSDF volume representation
   */
  void Init(const TsdfVolumeOptions &options);

  /**
   * @brief      Frees the memory used by the class.
   */
  void Free();

  /**
   * @brief      Gets the voxel at the specified world position.
   *
   * @param[in]  position  The 3D position in world coordinates
   *
   * @return     The voxel at the given position.
   */
  __host__ __device__ Voxel GetVoxel(float3 position);

  /**
   * @brief      Gets the voxel at the specified world position obtained using 
   *             trilinear interpolation.
   *
   * @param[in]  position  The 3D position in world coordinates
   *
   * @return     The voxel containing the interpolated values.
   */
  __host__ __device__ Voxel GetInterpolatedVoxel(float3 position);

  /**
   * @brief      Sets the voxel at the specified position to the specified
   *             values.
   *
   * @param[in]  position  The 3D position in world coordinates
   * @param[in]  voxel     The voxel containing the values to be set
   *
   * @return     True if the voxel was successfully set. False if the voxel was
   *             not found (it is not allocated).
   */ 
  __host__ __device__ bool SetVoxel(float3 position, const Voxel& voxel);

  /**
   * @brief      Updates the voxel at the given position by computing a weighted
   *             average with the given voxel.
   *
   * @param[in]  position  The 3D position in world coordinates
   * @param[in]  voxel     The new voxel used to update the one in the volume
   *
   * @return     True if the voxel was successfully set. False if the voxel was
   *             not found (it is not allocated).
   */
  __host__ __device__ bool UpdateVoxel(float3 position, const Voxel& voxel);

  /**
   * @brief      Generates a virtual depth image using raycasting from the given
   *             camera pose.
   *
   * @param[in]  camera_pose  The camera pose
   * @param[in]  sensor       The intrinsic parameters of the sensor
   *
   * @return     The virtual depth image (in meters).
   */
  float* GenerateDepth(float4x4 camera_pose, RgbdSensor sensor);

  /**
   * @brief      Generates a virtual RGB image using raycasting from the given
   *             camera pose.
   *
   * @param[in]  camera_pose  The camera pose
   * @param[in]  sensor       The intrinsic parameters of the sensor
   *
   * @return     The virtual RGB image (each channel in the range 0-255).
   */
  uchar3* GenerateRgb(float4x4 camera_pose, RgbdSensor sensor);

  /**
   * @brief      Extracts a mesh from the portion of the volume within the
   *             specified bounding box.
   *
   * @param[in]  lower_corner  The lower corner of the bounding box
   * @param[in]  upper_corner  The upper corner of the bounding box
   *
   * @return     The mesh.
   */
  Mesh ExtractMesh(const float3 &lower_corner, const float3 &upper_corner);

  /**
   * @brief      Integrates a new RGB-D scan into the volume, given the pose of
   *             the sensor. It ignores masked values.
   *
   * @param[in]  image        The RGB-D image
   * @param[in]  camera_pose  The camera pose
   * @param      mask         A boolean mask of the same size as the image. True
   *                          values are ignored, i.e., are not integrated
   */
  void IntegrateScan(const RgbdImage &image, float4x4 camera_pose, bool *mask);

  /**
   * @brief      Gets the options for the TSDF representation.
   *
   * @return     The options.
   */
  __host__ __device__ TsdfVolumeOptions GetOptions();

 protected:
  /**
   * @brief      Converts coordinates from global voxel indices to world 
   *             coordinates (in meters).
   *
   * @param[in]  position  The global voxel position
   *
   * @return     The world coordinates (in meters).
   */
  __host__ __device__ float3 GlobalVoxelToWorld(int3 position);

  /**
   * @brief      Converts coordinates from world coordinates (in meters) to
   *             global voxel indices.
   *
   * @param[in]  position  The position in world coordinates
   *
   * @return     The global voxel position.
   */
  __host__ __device__ int3 WorldToGlobalVoxel(float3 position);

  /**
   * @brief      Converts coordinates from world coordinates (in meters) to
   *             blocks coordinates (indices)
   *
   * @param[in]  position  The position in world coordinates
   *
   * @return     The block position.
   */
  __host__ __device__ int3 WorldToBlock(float3 position);

  /**
   * @brief      Converts coordinates from world coordinates (in meters) to
   *             local indices of the voxel within its block
   *
   * @param[in]  position  The position in world coordinates
   *
   * @return     The local indices of the voxel within its block.
   */
  __host__ __device__ int3 WorldToLocalVoxel(float3 position);

  /** Options for the TSDF representation */
  TsdfVolumeOptions options_;
};

}  // namespace tsdfvh

}  // namespace refusion
