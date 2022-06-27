#include "Simulation.cuh"
#include <cuda_runtime.h>
#include <FluidEngine/Compute/Utility/CUDAGLInterop.h>

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
			cudaThreadSynchronize();
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
			cudaThreadSynchronize();
		}
	}
}