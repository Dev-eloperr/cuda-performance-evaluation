
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "VectorAdditionBenchmarker.h"

int main() {
	VectorAdditionBenchmarker vectorAddition(40000, 10000);
	std::cout << "Running Vector Addition Benchmarks-------------------" << std::endl;
	vectorAddition.benchmark();
}
