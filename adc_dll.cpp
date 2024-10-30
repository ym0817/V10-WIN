#include "adc_dll.h"
#include "adc.h"
#include <iostream>
#include <vector>
using namespace std;

Detector* g_singleDetector;

bool LoadADCModel(char* cfgPath, char* weightPath, char* nameList, int threadNum)
{
	bool status(true);

	std::string str(cfgPath);
	int isgpu = std::stoi(str); 

	g_singleDetector = new Detector(weightPath, isgpu, nameList, threadNum);

	//status = LoadADCModel_Main(cfgPath, weightPath, nameList, threadNum);

	return status;
}


bool ADCModelInferenceImage(char* imgPath, float scoreTh, int& objNum, int** classID, float** score)
{
	bool status(true);

	cv::Mat mat_img = cv::imread(imgPath);
	cv::Mat out_mat_img = mat_img.clone();
	int orig_width = mat_img.cols;
	int orig_height = mat_img.rows;

	std::vector<float> input_tensor_values = g_singleDetector->preprocessImage(mat_img);
	auto start = std::chrono::steady_clock::now();
	std::vector<float> results = g_singleDetector->runInference(input_tensor_values);
	std::vector<Detection> detections = g_singleDetector->filterDetections(results, scoreTh, g_singleDetector->input_shape[2], g_singleDetector->input_shape[3], orig_width, orig_height);
	auto end = std::chrono::steady_clock::now();
	//std::chrono::duration<double> spent = end - start;
	double spent_ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << " ----------------------- Time: " << spent_ms << " ms \n";
	std::cout << " -----------detections Num : " << detections.size() << "  \n";


	cv::Mat output = g_singleDetector->draw_labels(out_mat_img, detections);
	//the result file will be saved in the build folder:
	//cv::imwrite("result.jpg", output);
	//cv::imshow("output", output);
	//cv::waitKey(0);


	objNum = detections.size();

	*classID = new int[objNum];
	*score = new float[objNum];

	for (int i = 0; i < detections.size(); i++)
	{
		(*classID)[i] = detections[i].class_id;
		(*score)[i] = detections[i].confidence;
	}


	return status;
}
