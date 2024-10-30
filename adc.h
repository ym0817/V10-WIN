#pragma once
#ifndef IADC_H
#define IADC_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


struct Detection
{
    float confidence;
    cv::Rect bbox;
    int class_id;
    std::string class_name;
};


class Detector {
public:
    Detector(const std::string& model_path, int isGPU, std::string namesList, int threadNum);
    ~Detector();


    std::vector<float> preprocessImage(const cv::Mat& image);
    std::vector<Detection> filterDetections(const std::vector<float>& results, float confidence_threshold, int img_width, int img_height, int orig_width, int orig_height);
    std::vector<float> runInference(const std::vector<float>& input_tensor_values);

    cv::Mat draw_labels(const cv::Mat& image, const std::vector<Detection>& detections);
    std::vector<int64_t> input_shape;





private:

    Ort::Env env{ nullptr };
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };
    Ort::AllocatorWithDefaultOptions allocator;


    std::string input_name;
    std::string output_name;
    std::vector<std::string> g_name_list;



};


#endif // IADC_H

