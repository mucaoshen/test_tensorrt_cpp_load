#include <string>
#include "test.h"


int main() {
	std::string modelPath("../ResNet34_trackerOCR_36_450_20230627_half.engine");
	B* a = new A();
	int retCode = a->modelLoad(modelPath);
}