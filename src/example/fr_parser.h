// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#pragma once
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>

/**
 * @brief      Simple parser for data in the TUM RGB-D Benchmark format.
 */
class FrParser {
 public:
  /**
   * @brief      Constructs the class.
   *
   * @param[in]  start_frame  The start frame
   * @param[in]  end_frame    The end frame (-1 = until the end of the file)
   * @param[in]  frame_skip   The frame skip (1 = loads every image, 2 = loads 
   *                          every other image, etc.)
   */
  FrParser(int start_frame = 0, int end_frame = -1,
           int frame_skip = 1);

  /**
   * @brief      Opens the associated.txt file.
   *
   * @param[in]  filepath  The path of the folder of the dataset
   *
   * @return     True if the file was successfully opened. False otherwise.
   */
  bool OpenFile(std::string filepath);

  /**
   * @brief      Gets the current RGB image.
   *
   * @return     The current RGB image.
   */
  cv::Mat GetRgb();

  /**
   * @brief      Gets the current depth image.
   *
   * @return     The current depth image.
   */
  cv::Mat GetDepth();

  /**
   * @brief      Gets the timestamp of the current RGB image.
   *
   * @return     The timestamp of the current RGB image.
   */
  double GetTimestampRgb();

  /**
   * @brief      Gets the timestamp of the current depth image.
   *
   * @return     The timestamp of the current depth image.
   */
  double GetTimestampDepth();

  /**
   * @brief      Gets the index of the current image.
   *
   * @return     The index of the current image.
   */
  int GetIndex();

  /**
   * @brief      Reads the next line of associated.txt and loads the respective
   *             RGB and depth images.
   *
   * @return     True if the images were successfully loaded. False if the file
   *             associated.txt has ended or was never loaded.
   */
  bool ReadNext();

 protected:
  /** Stores whether the file associated.txt is open or not */
  bool file_opened_ = false;

  /** The index of the current image */
  int index_ = 0;

  /** The file associated.txt */
  std::ifstream fassociated_;

  /** The path of the dataset */
  std::string filebase_;

  /** The desired initial frame */
  int start_frame_;

  /** The desired final frame */
  int end_frame_;

  /** The frame skip */
  int frame_skip_;

  /** The current RGB image */
  cv::Mat rgb_;

  /** The current depth image */
  cv::Mat depth_;

  /** The timestamp of the current RGB image */
  double timestamp_rgb_;

  /** The timestamp of the current depth image */
  double timestamp_depth_;
};
