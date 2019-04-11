#pragma once
#include <chrono>

#include "Benchmarker.h"

class VectorAdditionBenchmarker : public Benchmarker
{
public:
	VectorAdditionBenchmarker(size_t size, size_t arraySize);

	std::chrono::high_resolution_clock::duration runGpu() override;
	std::chrono::high_resolution_clock::duration runCpu() override;
	
private:
	size_t arraySize;
};
