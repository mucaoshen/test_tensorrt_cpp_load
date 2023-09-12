#include <string>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#define CHECK(call, resContent) check(call, __LINE__, __FILE__, resContent)

inline bool check(cudaError_t e, int iLine, const char *szFile, std::string& resContent) {
	if (e != cudaSuccess) {
		resContent = "CUDA runtime API error ";
		resContent += std::string(cudaGetErrorName(e));
		resContent += " at line " + std::to_string(iLine);
		resContent += " in file " + std::string(szFile);
		resContent += "\n";
		// std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
	}
	resContent = "";
	return true;
};

class TRTLogger: public nvinfer1::ILogger {
public:
	nvinfer1::ILogger::Severity reportableServerity;

public:
	TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kVERBOSE): reportableServerity(severity) {
	}
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
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
	};
};
static TRTLogger s_Logger = TRTLogger();


int modelLoad(const std::string& m_modelPath) {
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

	std::vector<char> engineStr(fsize);
	engineFile.read(engineStr.data(), engineStr.size());

	if (engineStr.size() == 0) {
		std::cout<<"Failed getting serialized engine!"<<std::endl;
		engineFile.close();
		return -1;
	}
	engineFile.close();
	std::cout<<"Succeeded getting serialized engine!"<<std::endl;

	// create inference env, deserialize engine
	nvinfer1::IRuntime* m_runtime {nvinfer1::createInferRuntime(s_Logger)};
	nvinfer1::ICudaEngine* m_engine = m_runtime->deserializeCudaEngine(engineStr.data(), engineStr.size());
	if (m_engine == nullptr) {
		std::cout<<"Failed loading engine!"<<std::endl;
		return -1;
	}
	return 0;
}

int main() {
	std::string modelPath("../ResNet34_trackerOCR_36_450_20230627_half.engine");
	int retCode = modelLoad(modelPath);
}