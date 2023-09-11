#include "test.h"
#include <iostream>
#include <fstream>
#include <vector>



void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
	if (severity > reportableServerity) {
		return;
	}
	switch (severity)
	{
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
		std::cout<<"INTERNAL_ERROR: " + std::string(msg)<<std::endl;
		break;

	case nvinfer1::ILogger::Severity::kERROR:
		std::cout<<"ERROR: " + std::string(msg)<<std::endl;
		break;

	case nvinfer1::ILogger::Severity::kWARNING:
		std::cout<<"WARNING: " + std::string(msg)<<std::endl;
		break;

	case nvinfer1::ILogger::Severity::kINFO:
		std::cout<<"INFO: " + std::string(msg)<<std::endl;
		break;
	
	default:
		std::cout<<"VERBOSE: " + std::string(msg)<<std::endl;
		break;
	}
}

TRTLogger A::s_Logger = TRTLogger();
int A::modelLoad(const std::string& m_modelPath) {
	std::string tmpLogStr;
	bool isSuccess = CHECK(cudaSetDevice(0), tmpLogStr);
	if (!isSuccess) {
		throw std::runtime_error("cuda set device in modelLoad unsuccessfully : " + tmpLogStr);
	}

	std::ifstream engineFile(m_modelPath, std::ios::binary);
	long int fsize = 0;
	// get file size
	std::cout<<"Parsing model file!"<<std::endl;
	engineFile.seekg(0, engineFile.end);
	fsize = engineFile.tellg();
	engineFile.seekg(0, engineFile.beg);
	// get meta info
	char* metaLenBytes;
	metaLenBytes = (char*)malloc(4);
	engineFile.read(metaLenBytes, 4);
	int metaLen = bytesToInteger(metaLenBytes);
	if (metaLenBytes != nullptr) free(metaLenBytes);
	// TODO: get meta json str
	engineFile.seekg(4, engineFile.beg);
	char* metaBytes;
	metaBytes = (char*)malloc(metaLen);
	engineFile.read(metaBytes, metaLen);
	if (metaBytes != nullptr) free(metaBytes);
	// get model info
	std::vector<char> engineStr(fsize - metaLen - 4);
	engineFile.seekg(metaLen + 4, engineFile.beg);
	engineFile.read(engineStr.data(), fsize - metaLen - 4);

	if (engineStr.size() == 0) {
		std::cout<<"Failed getting serialized engine!"<<std::endl;
		engineFile.close();
		return -1;
	}
	engineFile.close();
	std::cout<<"Succeeded getting serialized engine!"<<std::endl;

	// create inference env, deserialize engine
	nvinfer1::IRuntime* m_runtime {nvinfer1::createInferRuntime(s_Logger)};
	m_engine = m_runtime->deserializeCudaEngine(engineStr.data(), engineStr.size());
	if (m_engine == nullptr) {
		std::cout<<"Failed loading engine!"<<std::endl;
		return -1;
	}
	return 0;
}