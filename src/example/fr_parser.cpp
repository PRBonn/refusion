// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#include "example/fr_parser.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
FrParser::FrParser(int start_frame, int end_frame,
                   int frame_skip)
    : start_frame_(start_frame),
      end_frame_(end_frame),
      frame_skip_(frame_skip) {}

bool FrParser::OpenFile(std::string filepath) {
  filebase_ = filepath;
  filebase_ += '/';
  std::stringstream assoc_path;
  assoc_path << filebase_ << "/associated.txt";
  fassociated_.open(assoc_path.str());
  if (fassociated_.is_open()) {
    file_opened_ = true;
    return true;
  } else {
    return false;
  }
}

cv::Mat FrParser::GetRgb() { return rgb_; }

cv::Mat FrParser::GetDepth() { return depth_; }

double FrParser::GetTimestampRgb() { return timestamp_rgb_; }

double FrParser::GetTimestampDepth() { return timestamp_depth_; }

int FrParser::GetIndex() { return index_ - 1; }

bool FrParser::ReadNext() {
  if (file_opened_) {
    std::string filepath_depth, filepath_rgb;
    while (index_ < start_frame_ || index_ % frame_skip_ != 0) {
      fassociated_ >> timestamp_depth_ >> filepath_depth >> timestamp_rgb_ >>
          filepath_rgb;
      index_++;
    }

    if (fassociated_ >> timestamp_depth_ >> filepath_depth >>
                        timestamp_rgb_ >> filepath_rgb &&
                    (index_ < end_frame_ || end_frame_ == -1)) {
      std::stringstream filepath;
      filepath << filebase_ << filepath_rgb;
      rgb_ = cv::imread(filepath.str(), CV_LOAD_IMAGE_COLOR);
      filepath.str("");
      filepath.clear();
      filepath << filebase_ << filepath_depth;
      depth_ = cv::imread(filepath.str(), CV_LOAD_IMAGE_ANYDEPTH);
      depth_.convertTo(depth_, CV_32FC1, 1.0f / 5000);
      index_++;
      return true;
    }
  }
  return false;
}
