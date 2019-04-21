#include "LinearSearch.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <math.h>
#include <memory>

namespace {
	__global__
		void searchWithCuda(int64_t * arr, size_t N, int64_t x) {
		int index = threadIdx.x;
		int stride = blockDim.x;

		for (int i = index; i < N; i += stride) {
			if (arr[i] == x) {
				break;
			}
		}
	}
}


LinearSearch::LinearSearch(size_t size, size_t arraySize) : arraySize(arraySize), Benchmarker(size)
{
}


LinearSearch::~LinearSearch()
{
}

std::chrono::high_resolution_clock::duration LinearSearch::runCpu()
{
	auto array = std::make_unique<int64_t[]>(arraySize);
	std::mt19937_64 rand;
	for (auto i = 0; i < arraySize; i++) {
		array.get()[i] = rand();
	}
	int randIndex = abs(static_cast<long>(rand())) % arraySize;
	auto x = array.get()[randIndex];
	auto ptr = array.get();
	auto startTime = std::chrono::high_resolution_clock::now();
	for (auto i = 0; i < arraySize; i++) {
		if (ptr[i] == x) {
			break;
		}
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	return endTime - startTime;
}

std::chrono::high_resolution_clock::duration LinearSearch::runGpu()
{
	auto arr = std::make_unique<int64_t[]>(arraySize);
	std::mt19937_64 rand;
	for (auto i = 0; i < arraySize; i++) {
		arr.get()[i] = rand();
	}

	int64_t* gpuArray;
	cudaMalloc(&gpuArray, sizeof(int64_t) * arraySize);
	cudaMemcpy(gpuArray, arr.get(), sizeof(int64_t) * arraySize, cudaMemcpyHostToDevice);
	size_t threadsPerBlock, blocksPerGrid;
	if (arraySize < 512) {
		threadsPerBlock = arraySize;
		blocksPerGrid = 1;
	}
	else {
		threadsPerBlock = 512;
		blocksPerGrid = (size_t)ceil(double(arraySize) / double(threadsPerBlock));
	}
	int randIndex = std::abs(static_cast<long>(rand())) % arraySize;
	cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();
	::searchWithCuda << <blocksPerGrid, threadsPerBlock >> > (gpuArray, arraySize, arr.get()[randIndex]);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	cudaFree(gpuArray);
	return end - start;
}
