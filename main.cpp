#include "adc.h"
#include "adc_dll.h"
#include <iostream>
#include <opencv2/opencv.hpp>





int main()
{


	std::string cfgPath = "1";

	const std::string weightPath = "best.onnx";
	std::string nameList = "class.names";
	int	threadNum = 1;

	LoadADCModel((char*)cfgPath.c_str(), (char*)weightPath.c_str(), (char*)nameList.c_str(), threadNum);


	std::string ImgPath = "192_r.JPG";
	//std::string ImgPath = "E:\\v10\\ADC_Demo\\ADC_Demo\\185-FAB-41.JPG";
	float ConfTH = 0.7;
	int objNum = 0;
	int* ClassIDs = nullptr;
	float* Sscore = nullptr;

	for (int n = 0; n < 10; n++)
	{

		auto start = std::chrono::steady_clock::now();
		ADCModelInferenceImage((char*)ImgPath.c_str(), ConfTH, objNum, &ClassIDs, &Sscore);
		auto end = std::chrono::steady_clock::now();
		//std::chrono::duration<double> spent = end - start;
		double spent_ms = std::chrono::duration<double, std::milli>(end - start).count();
		std::cout << "---------PerImage cost time:    " << spent_ms << "   ms " << std::endl;
		for (int i = 0; i < objNum; i++)
		{
			std::cout << "ClassID " << ClassIDs[i] << ", score " << Sscore[i] << std::endl;
	
		}
	}

	// ADCModelInferenceImage((char*)ImgPath.c_str(), ConfTH, objNum, &ClassIDs, &Sscore);

	// for (int i = 0; i < objNum; i++)
	// {
	//	std::cout << "ClassID " << ClassIDs[i] << ", score " << Sscore[i] << std::endl;

	} // 


}
