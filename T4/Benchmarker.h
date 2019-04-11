#pragma once
#include <chrono>

class Benchmarker
{
public:
	Benchmarker(int);
	~Benchmarker();

	virtual std::chrono::high_resolution_clock::duration runGpu() = 0;
	virtual std::chrono::high_resolution_clock::duration runCpu() = 0;
	void benchmark();
private:
	unsigned int benchmarkSize;
};

