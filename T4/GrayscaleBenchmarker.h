#include "Benchmarker.h"

class GrayscaleBenchmarker : public Benchmarker {
public:
	GrayscaleBenchmarker(size_t, size_t);
	
	std::chrono::high_resolution_clock::duration runGpu() override;
	std::chrono::high_resolution_clock::duration runCpu() override;
};

