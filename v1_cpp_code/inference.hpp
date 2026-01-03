#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

class YoloDetector {
public:
    YoloDetector(const std::string& modelPath);
    std::vector<Detection> detect(cv::Mat& frame);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    
    std::vector<const char*> inputNames = {"images"};
    std::vector<const char*> outputNames = {"output0"};

    const float confThreshold = 0.40f;
    const float nmsThreshold = 0.45f;
    const cv::Size inputSize = cv::Size(640, 640);
};

#endif