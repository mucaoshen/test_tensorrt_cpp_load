#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>

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
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

};

class B {
	public:
	virtual int modelLoad(const std::string& m_modelPath) = 0;

};

class A: public B {
	public:
	int modelLoad(const std::string& m_modelPath) override;
	static TRTLogger s_Logger;
	private:
	nvinfer1::ICudaEngine* m_engine;
};

inline int bytesToInteger(char* buffer) {
	return *reinterpret_cast<int*>(buffer);
}
