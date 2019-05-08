// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "example/fr_parser.h"
#include "tracker/tracker.h"
int main(int argc, char const *argv[]) {
  if (argc < 1) {
    std::cerr << "Usage: fb_data_parser <dataset_path>" << std::endl;
    return -1;
  }

  // Options for the TSDF representation
  refusion::tsdfvh::TsdfVolumeOptions tsdf_options;
  tsdf_options.voxel_size = 0.01;
  tsdf_options.num_buckets = 50000;
  tsdf_options.bucket_size = 10;
  tsdf_options.num_blocks = 500000;
  tsdf_options.block_size = 8;
  tsdf_options.max_sdf_weight = 64;
  tsdf_options.truncation_distance = 0.1;
  tsdf_options.max_sensor_depth = 5;
  tsdf_options.min_sensor_depth = 0.1;

  // Options for the tracker
  refusion::TrackerOptions tracker_options;
  tracker_options.max_iterations_per_level[0] = 6;
  tracker_options.max_iterations_per_level[1] = 3;
  tracker_options.max_iterations_per_level[2] = 2;
  tracker_options.downsample[0] = 4;
  tracker_options.downsample[1] = 2;
  tracker_options.downsample[2] = 1;
  tracker_options.min_increment = 0.0001;
  tracker_options.regularization = 0.002;
  tracker_options.huber_constant = 0.02;

  // Intrinsic parameters of the sensor
  refusion::RgbdSensor sensor;
  sensor.cx = 319.5f;
  sensor.cy = 239.5f;
  sensor.fx = 525.0;
  sensor.fy = 525.0;
  sensor.rows = 480;
  sensor.cols = 640;
  sensor.depth_factor = 5000;

  refusion::Tracker tracker(tsdf_options, tracker_options, sensor);

  std::string filebase(argv[1]);
  filebase += '/';
  std::stringstream filepath_out, filepath_time;
  filepath_out << filebase << "/result.txt";
  std::ofstream result(filepath_out.str());

  int start = 0;
  int end = -1;
  int frame_skip = 1;
  FrParser parser(start, end, frame_skip);
  if (parser.OpenFile(filebase)) {
    while (parser.ReadNext()) {
      tracker.AddScan(parser.GetRgb(), parser.GetDepth());
      Eigen::Matrix4d pose = tracker.GetCurrentPose();
      Eigen::Quaterniond rotation(pose.block<3, 3>(0, 0));
      result << std::fixed << std::setprecision(6) << parser.GetTimestampDepth()
             << " " << pose.block<3, 1>(0, 3).transpose() << " "
             << rotation.vec().transpose() << " " << rotation.w() << std::endl;
      std::cout << parser.GetIndex() << std::endl;
      std::cout << tracker.GetCurrentPose() << std::endl;

      cv::Mat virtual_rgb = tracker.GenerateRgb(1280, 960);
      cv::imshow("RGB", virtual_rgb);
      int k = cv::waitKey(1);
      if (k == 27) {
        break;
      }
    }
  } else {
    std::cerr << "Error while opening file associated.txt" << std::endl;
    return -1;
  }

  result.close();

  std::cout << "Creating mesh..." << std::endl;
  float3 low_limits = make_float3(-3, -3, 0);
  float3 high_limits = make_float3(3, 3, 4);
  refusion::tsdfvh::Mesh *mesh;
  cudaMallocManaged(&mesh, sizeof(refusion::tsdfvh::Mesh));
  *mesh = tracker.ExtractMesh(low_limits, high_limits);
  filepath_out.str("");
  filepath_out.clear();
  filepath_out << filebase << "mesh.obj";
  mesh->SaveToFile(filepath_out.str());
  return 0;
}
