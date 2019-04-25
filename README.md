# cuda-performance-evaluation
## Computer Architecture (CSD208) Project
The project aims to find performance improvements of commonly used algorithms on graphics processing units (GPUs)
using General Purpose Programming on GPUs (GPGPUs). We
have used Nvidia’s CUDA library for programming. Having
ran algorithms of various time complexities and parallelizability, we were able to observe interesting behaviours of these
brilliant devices.

We ran our benchmarks on two machines, a Lenovo Gaming
Laptop with an Nvidia GTX 1050, and a workstation-class
Nvidia K80 cloud-hosted on an Amazon Web Services (AWS)
virtualized server. We had initially planned to run on the GPU
cluster on our university’s high performance computer, Magus,
however, due to unavailability of the GPU cluster, we had to
abandon this plan.

To obtain fast performance without the overhead of garbage
collection and dynamic memory allocation, we wrote all of
our benchmarks using C++14. All the code was compiled
using Nvidia’s CUDA Compiler (NVCC) which is based on the popular open-source optimizing Clang LLVM compiler.

### References
- [Cuda Documentation](https://docs.nvidia.com/cuda/)
- [fttw3 Documentation](http://www.fftw.org/)
