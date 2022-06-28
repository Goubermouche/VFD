#include "Simulation.cuh"
#include <cuda_runtime.h>
#include <FluidEngine/Compute/Utility/CUDAGLInterop.h>
#include <iostream>
#include <FluidEngine/Compute/Utility/cutil.h>

namespace fe {
	extern "C" {
		void SetParameters(SimulationParameters* params) {
			cudaMemcpyToSymbol(parameters, params, sizeof(SimulationParameters));
		}

		int IDivUp(int a, int b) {
			return a % b != 0 ? a / b + 1 : a / b;
		}

		void ComputeGridSize(int n, int blockSize, int& blockCount, int& threadCount) {
			threadCount = min(blockSize, n);
			blockCount = IDivUp(n, threadCount);
		}

		void CalculateHash(uint vboPosition, uint2* particleHash, int particleCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			float4* position;
			cudaGLMapBufferObject((void**)&position, vboPosition);
			// kernel
			cudaGLUnmapBufferObject(vboPosition);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		}

		void Integrate(uint vboOldPosition, uint vboNewPosition, float4* oldVelocity, float4* newVelocity, int particleCount) {
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			float4* oldPosition;
			float4* newPosition;
			cudaGLMapBufferObject((void**)&oldPosition, vboOldPosition);
			cudaGLMapBufferObject((void**)&newPosition, vboNewPosition);
			IntegrateKernel <<< blockCount, threadCount >>> (oldPosition, newPosition, oldVelocity, newVelocity);
			cudaGLUnmapBufferObject(vboOldPosition);
			cudaGLUnmapBufferObject(vboNewPosition);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		}

		void Reorder(uint vboOldPosition, float4* oldVelocity, float4* sortedPosition, float4* sortedVelocity, uint2* particleHash, uint* cellStart, uint particleCount, uint cellCount)
		{
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			float4* oldPosition;
			cudaGLMapBufferObject((void**)&oldPosition, vboOldPosition);

#if USE_TEX
			uint particleMemorySize4 = particleCount * sizeof(float4);
			cudaBindTexture(0, oldPositionTexture, oldPosition, particleMemorySize4);
			cudaBindTexture(0, oldVelocityTexture, oldVelocity, particleMemorySize4);
#endif // USE_TEX
			ReorderKernel <<< blockCount, threadCount >>> (particleHash, cellStart, oldPosition, oldVelocity, sortedPosition, sortedVelocity);
#if USE_TEX
			cudaUnbindTexture(oldPositionTexture);
			cudaUnbindTexture(oldVelocityTexture);
#endif  // USE_TEX
			cudaGLUnmapBufferObject(vboOldPosition);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		}

		void Collide(uint vboOldPosition, uint vboNewPosition, float4* sortedPosition, float4* sortedVelocity, float4* oldVelocity, float4* newVelocity, float* pressure, float* density, uint2* particleHash, uint* cellStart, uint particleCount, uint cellCount)
		{
			float4* oldPosition;
			float4* newPosition;
			
			cudaGLMapBufferObject((void**)&oldPosition, vboOldPosition);
			cudaGLMapBufferObject((void**)&newPosition, vboNewPosition);

#if USE_TEX
			uint particleMemorySize4 = particleCount * sizeof(float4);
			uint particleMemorySize1 = particleCount * sizeof(float);
			cudaBindTexture(0, oldPositionTexture, sortedPosition, particleMemorySize4);
			cudaBindTexture(0, oldVelocityTexture, sortedVelocity, particleMemorySize4);
			cudaBindTexture(0, pressureTexture, pressure, particleMemorySize1);
			cudaBindTexture(0, densityTexture, density, particleMemorySize1);
			cudaBindTexture(0, particleHashTexture, particleHash, particleCount * sizeof(uint2));
			cudaBindTexture(0, cellStartTexture, cellStart, cellCount * sizeof(uint));
#endif // USE_TEX

			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			CalculateDensityKernel <<< blockCount, threadCount >>> (sortedPosition, pressure, density, particleHash, cellStart);
			CUDA_SAFE_CALL(cudaThreadSynchronize());

			CalculateForceKernel <<< blockCount, threadCount >>> (newPosition, newVelocity, sortedPosition, sortedVelocity, pressure, density, particleHash, cellStart);
			
			cudaGLUnmapBufferObject(vboNewPosition);
			cudaGLUnmapBufferObject(vboOldPosition);

#if USE_TEX
			cudaUnbindTexture(oldPositionTexture);
			cudaUnbindTexture(oldVelocityTexture);
			cudaUnbindTexture(pressureTexture);
			cudaUnbindTexture(densityTexture);
			cudaUnbindTexture(particleHashTexture);
			cudaUnbindTexture(cellStartTexture);
#endif // USE_TEX
			CUDA_SAFE_CALL(cudaThreadSynchronize());
		}
	}
}