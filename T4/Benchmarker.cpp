#include "Benchmarker.h"
#include <iostream>

Benchmarker::Benchmarker(int size = 100) : benchmarkSize(size)
{
}


Benchmarker::~Benchmarker()
{
}

void Benchmarker::benchmark()
{
	std::chrono::high_resolution_clock::duration gpuTime{0};
	std::chrono::high_resolution_clock::duration cpuTime{0};

	for (size_t i = 0; i < benchmarkSize; i++) {
		gpuTime += runGpu();
	}

	for (size_t i = 0; i < benchmarkSize; i++) {
		cpuTime += runCpu();
	}

	std::cout << "Average time taken by GPU: " <<
		std::chrono::duration_cast<std::chrono::microseconds>(gpuTime).count() / static_cast<double>(benchmarkSize)
		<< " us." << std::endl;	
	
	std::cout << "Average time taken by CPU: " <<
		std::chrono::duration_cast<std::chrono::microseconds>(cpuTime).count() / static_cast<double>(benchmarkSize)
		<< " us." << std::endl;
}
