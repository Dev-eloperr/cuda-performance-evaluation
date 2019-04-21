#!/bin/bash
set -euxo pipefail

sudo apt-get update && sudo apt-get upgrade
sudo apt-get install fftw3 fftw3-dev
cd T4
nvcc *.cpp *.cu -I. -lm -lcufft -lfftw3 -std=c++14 -O2
./a.out


