#pragma once
#include "Benchmarker.h"
class LinearSearch :
	public Benchmarker
{
public:
	LinearSearch(size_t, size_t);
	~LinearSearch();

	std::chrono::high_resolution_clock::duration runCpu() override;
	std::chrono::high_resolution_clock::duration runGpu() override;
private:
	size_t arraySize;
};

