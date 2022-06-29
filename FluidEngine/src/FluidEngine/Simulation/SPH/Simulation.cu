#include "Simulation.cuh"
#include <cuda_runtime.h>
#include <FluidEngine/Compute/Utility/CUDAGLInterop.h>
#include <iostream>
#include <FluidEngine/Compute/Utility/cutil.h>

namespace fe {
	extern "C" {
		int IDivUp(int a, int b) {
			return a % b != 0 ? a / b + 1 : a / b;
		}

		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount) {
			threadCount = min(blockSize, n);
			blockCount = IDivUp(n, threadCount);
		}
	}
}