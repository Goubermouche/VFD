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

		void Integrate(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, int particleCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			IntegrateKernel <<< blockCount, threadCount >>> (newPos, newVel, oldPos, oldVel);
			CUT_CHECK_ERROR("Kernel execution failed: IntegrateKernel");
		}

		void CalculateHash(float4* pos, uint2* particleHash, int particleCount) 
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 2 * 256, blockCount, threadCount);

			CalculateHashKernel <<< blockCount, threadCount >>> (pos, particleHash);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateHashKernel");
		}

		void Reorder(uint2* particleHash, uint* cellStart, float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel, int particleCount, int cellCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);
			CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, cellCount * sizeof(uint)));
			ReorderKernel <<< blockCount, threadCount >>> (particleHash, cellStart, oldPos, oldVel, sortedPos, sortedVel);
			CUT_CHECK_ERROR("Kernel execution failed: ReorderKernel");
		}

		void CalculateDensity(float4* sortedPos, float* pressure, float* density, uint2* particleHash, uint* cellStart, int particleCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 64, blockCount, threadCount);
			CalculateDensityKernel <<< blockCount, threadCount >>> (sortedPos, pressure, density, particleHash, cellStart);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateDensityKernel");
		}

		void CalculateForce(float4* newPos, float4* newVel, float4* sortedPos, float4* sortedVel, float* pressure, float* density, uint2* particleHash, uint* cellStart, int particleCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 64, blockCount, threadCount);
			CalculateForceKernel <<< blockCount, threadCount >>> (newPos, newVel, sortedPos, sortedVel, pressure, density, particleHash, cellStart);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateForceKernel");
		}
	}
}