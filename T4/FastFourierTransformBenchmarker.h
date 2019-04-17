#pragma once
#include "Benchmarker.h"
class FastFourierTransformBenchmarker :
	public Benchmarker
{
public:
	FastFourierTransformBenchmarker(size_t, size_t);

	std::chrono::high_resolution_clock::duration runGpu() override;
	std::chrono::high_resolution_clock::duration runCpu() override;

	~FastFourierTransformBenchmarker();
private:
	size_t arraySize;
};

