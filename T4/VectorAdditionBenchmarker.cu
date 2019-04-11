#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <random>
#include <memory>

#include "VectorAdditionBenchmarker.h"

namespace {

	__global__ void kernel_sum(const int64_t* a, const int64_t* b, int64_t* sum, const size_t size) {
		size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid < size) {
			sum[tid] = a[tid] + b[tid];
		}
	}

	std::chrono::high_resolution_clock::duration sum(const int64_t* a, const int64_t* b, int64_t* sum, const size_t size) {
		size_t threadsPerBlock, blocksPerGrid;
		if (size < 512) {
			threadsPerBlock = size;
			blocksPerGrid = 1;
		}
		else {
			threadsPerBlock = 512;
			blocksPerGrid = ceil(double(size) / double(threadsPerBlock));
		}

		auto startTime = std::chrono::high_resolution_clock::now();
		::kernel_sum <<<blocksPerGrid, threadsPerBlock>>> (a, b, sum, size);
		auto endTime = std::chrono::high_resolution_clock::now();
		return endTime - startTime;
	}
}

std::chrono::high_resolution_clock::duration VectorAdditionBenchmarker::runGpu()
{
	const size_t SIZE = arraySize;
	std::mt19937_64 random;
	auto a = std::make_unique<int64_t[]>(SIZE);
	auto b = std::make_unique<int64_t[]>(SIZE);
	auto c = std::make_unique<int64_t[]>(SIZE);

	for (size_t i = 0; i < SIZE; i++) {
		a.get()[i] = random();
		b.get()[i] = random();
	}

	int64_t* gpu_a;
	int64_t* gpu_b;
	int64_t* gpu_c;

	cudaSetDevice(0);

	cudaMalloc((void**)&gpu_a, SIZE * sizeof(int64_t));
	cudaMalloc((void**)&gpu_b, SIZE * sizeof(int64_t));
	cudaMalloc((void**)&gpu_c, SIZE * sizeof(int64_t));

	cudaMemcpy(gpu_a, a.get(), SIZE * sizeof(int64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b.get(), SIZE * sizeof(int64_t), cudaMemcpyHostToDevice);

	auto time = sum(gpu_a, gpu_b, gpu_c, SIZE);
	cudaMemcpy(c.get(), gpu_c, SIZE * sizeof(int64_t), cudaMemcpyDeviceToHost);

	if (!(a.get()[SIZE / 2] - b.get()[SIZE / 2])) {
		throw std::runtime_error("Unexpected behaviour. CUDA sum calculation is not correct.");
	}

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	return time;
}

std::chrono::high_resolution_clock::duration VectorAdditionBenchmarker::runCpu()
{
	const size_t SIZE = arraySize;
	std::mt19937_64 random;
	auto a = std::make_unique<int64_t[]>(SIZE);
	auto b = std::make_unique<int64_t[]>(SIZE);
	auto c = std::make_unique<int64_t[]>(SIZE);

	for (size_t i = 0; i < SIZE; i++) {
		a.get()[i] = random();
		b.get()[i] = random();
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < SIZE; i++) {
		c[i] = a[i] + b[i];
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	return endTime - startTime;
}

VectorAdditionBenchmarker::VectorAdditionBenchmarker(size_t size, size_t arraySize)
	: arraySize(arraySize), Benchmarker(size)
{
}