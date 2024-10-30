#include "adc.h"
#include <algorithm>
#include <iostream>
#include <fstream>


std::vector<std::string> objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    //std::cout << "object names loaded \n";
    return file_lines;
}



Detector::Detector(const std::string& model_path, int isGPU, std::string namesList, int threadNum)
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ADC");
    sessionOptions = Ort::SessionOptions();
    wchar_t path[1024];
    mbstowcs_s(nullptr, path, 1024, model_path.c_str(), 1024);
    session = Ort::Session(env, path, sessionOptions);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> net_input_shape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    input_shape.assign(net_input_shape.begin(), net_input_shape.end());
    Ort::AllocatedStringPtr in_name_allocator = session.GetInputNameAllocated(0, allocator);
    input_name = in_name_allocator.get();
    Ort::AllocatedStringPtr out_name_allocator = session.GetOutputNameAllocated(0, allocator);
    output_name = out_name_allocator.get();

    g_name_list = objects_names_from_file(namesList);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    //bool isGPU = false;
    //std::cout << "   isGPU   " << isGPU << std::endl;
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: CUDA GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    sessionOptions.SetIntraOpNumThreads(threadNum);
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

Detector::~Detector() {}



std::vector<float> Detector::preprocessImage(const cv::Mat& image)
{
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image");
    }

    cv::Mat resized_image, floatImage;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, resized_image, cv::Size(input_shape[2], input_shape[3]));
    resized_image.convertTo(floatImage, CV_32F, 1.0 / 255);

    /*  blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
      cv::Size floatImageSize{ floatImage.cols, floatImage.rows };*/
      //cv::imshow("resized_image", floatImage); //显示灰色图片
      //cv::waitKey(0);
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float*)channels[c].data, (float*)channels[c].data + input_shape[2] * input_shape[3]);
    }

    return input_tensor_values;
}


std::vector<float> Detector::runInference(const std::vector<float>& input_tensor_values)
{

    const char* input_name_ptr = input_name.c_str();
    const char* output_name_ptr = output_name.c_str();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_tensor_values.data()), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name_ptr, &input_tensor, 1, &output_name_ptr, 1);

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(floatarr, floatarr + output_tensor_size);
}




std::vector<Detection> Detector::filterDetections(const std::vector<float>& results, float confidence_threshold, int img_width, int img_height, int orig_width, int orig_height)
{
    std::vector<Detection> detections;
    const int num_detections = results.size() / 6;

    for (int i = 0; i < num_detections; ++i)
    {
        float left = results[i * 6 + 0];
        float top = results[i * 6 + 1];
        float right = results[i * 6 + 2];
        float bottom = results[i * 6 + 3];
        float confidence = results[i * 6 + 4];
        int class_id = results[i * 6 + 5];

        if (confidence >= confidence_threshold)
        {
            int x = static_cast<int>(left * orig_width / img_width);
            int y = static_cast<int>(top * orig_height / img_height);
            int width = static_cast<int>((right - left) * orig_width / img_width);
            int height = static_cast<int>((bottom - top) * orig_height / img_height);

            detections.push_back(
                { confidence,
                 cv::Rect(x, y, width, height),
                 class_id,
                 g_name_list[class_id] });
        }
    }

    return detections;
}





cv::Mat Detector::draw_labels(const cv::Mat& image, const std::vector<Detection>& detections)
{
    cv::Mat result = image.clone();

    for (const auto& detection : detections)
    {
        cv::rectangle(result, detection.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = detection.class_name + ": " + std::to_string(detection.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(
            result,
            cv::Point(detection.bbox.x, detection.bbox.y - labelSize.height),
            cv::Point(detection.bbox.x + labelSize.width, detection.bbox.y + baseLine),
            cv::Scalar(255, 255, 255),
            cv::FILLED);

        cv::putText(
            result,
            label,
            cv::Point(
                detection.bbox.x,
                detection.bbox.y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);

    }

    return result;
}


