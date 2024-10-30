#pragma once
#pragma once
#ifdef  ADCDLL_EXPORTS  
#define ADCDLL __declspec(dllexport)   
#else  
#define ADCDLL __declspec(dllimport)   
#endif  

extern "C"	ADCDLL bool LoadADCModel(char* cfgPath, char* weightPath, char* nameList, int threadNum);

//extern "C"	ADCDLL bool ADCTrainingModel(char* trainingFolder, int type);

extern "C"	ADCDLL bool ADCModelInferenceImage(char* imgPath, float scoreTh, int& objNum, int** classID, float** score);
