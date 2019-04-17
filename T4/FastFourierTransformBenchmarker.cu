#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "FastFourierTransformBenchmarker.h"
#include <cufft.h>
#include <fftw3.h>

FastFourierTransformBenchmarker::FastFourierTransformBenchmarker(size_t size, size_t arraySize)
	: arraySize(arraySize), Benchmarker(size)
{
}


std::chrono::high_resolution_clock::duration FastFourierTransformBenchmarker::runGpu()
{
	cufftHandle plan;
	cufftComplex *data;
	const auto NX = arraySize;
	cudaMalloc(&data, sizeof(cufftComplex) * NX );

	cufftPlan1d(&plan, NX, CUFFT_C2C, 10);
	auto startTime = std::chrono::high_resolution_clock::now();
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	auto endTime = std::chrono::high_resolution_clock::now();
	cufftDestroy(plan);
	cudaFree(data);
	return endTime - startTime;
}

std::chrono::high_resolution_clock::duration FastFourierTransformBenchmarker::runCpu()
{
	fftw_complex *in, *out;
	fftw_plan p;
	const auto N = arraySize;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	auto startTime = std::chrono::high_resolution_clock::now();
	fftw_execute(p); /* repeat as needed */
	auto endTime = std::chrono::high_resolution_clock::now();
	fftw_destroy_plan(p);
	fftw_free(in); fftw_free(out);
	return endTime - startTime;
}


FastFourierTransformBenchmarker::~FastFourierTransformBenchmarker()
{
}
