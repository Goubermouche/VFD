#include "pch.h"
#include "GPUSDF.cuh"

#include "GPUSDFKernel.cu"

namespace vfd {
	extern "C" {
		void ComputeSDF(float* V, int V_size, int* F, int F_size, float* sdf, int D1, int D2, int D3, float grid_size, float* min_corner)
		{
			int blockSize = 512;
			int numBlocks = (D1 * D2 * D3 + blockSize - 1) / blockSize;

			ComputeSDFKernel << <numBlocks, blockSize >> > (V, V_size,
					F, F_size,
					sdf, D1, D2, D3, grid_size, min_corner);

			COMPUTE_CHECK("Kernel execution failed: IntegrateKernel");
			COMPUTE_SAFE(cudaDeviceSynchronize());
		}
	}
}