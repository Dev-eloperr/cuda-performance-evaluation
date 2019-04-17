
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <string>

#include "VectorAdditionBenchmarker.h"
#include "FastFourierTransformBenchmarker.h"

namespace {
	template <typename T>
	void runBenchmark(std::string benchmarkName) {
		std::vector <std::pair<size_t, size_t>> benchmarks;
		benchmarks.push_back(std::make_pair(10000, 500));
		benchmarks.push_back(std::make_pair(10000, 5000));
		benchmarks.push_back(std::make_pair(10000, 50000));
		benchmarks.push_back(std::make_pair(10000, 100000));
		for (auto& i : benchmarks) {
			std::cout << "Running " << benchmarkName << " with array size " << i.second << std::endl;
			T t(i.first, i.second);
			t.benchmark();
		}
	}
}

int main() {
	runBenchmark<VectorAdditionBenchmarker>("Vector addition");
	runBenchmark<FastFourierTransformBenchmarker>("Fast fourier transform");
}

