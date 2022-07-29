#include "pch.h"

#include "FluidEngine/Compute/Utility/RadixSort/RadixSort.cuh"
#include "FluidEngine/Simulation/SPH/SimulationKernel.cu"
#include "FluidEngine/Compute/Utility/CudaKernelUtility.cuh"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	extern "C" {
		void SetParameters(SimulationData& params) {
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_Description, &params, sizeof(SimulationData)));
		}

		void Integrate(uint16_t oldPositionVBO, uint16_t newPositionVBO, glm::vec3* oldVelocity, glm::vec3* newVelocity, int particleCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			// Buffer data
			glm::vec3* oldPosition;
			glm::vec3* newPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&newPosition, newPositionVBO));

			// Kernel
			IntegrateKernel <<< blockCount, threadCount >>> (newPosition, oldPosition, newVelocity, oldVelocity);
			COMPUTE_CHECK("Kernel execution failed: IntegrateKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(oldPositionVBO));
			COMPUTE_SAFE(cudaGLUnmapBufferObject(newPositionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void CalculateHash(uint16_t positionVBO, glm::uvec2* particleHash, int particleCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 512, blockCount, threadCount);

			// Buffer data
			glm::vec3* position;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&position, positionVBO));

			// Kernel
			CalculateHashKernel <<< blockCount, threadCount >>> (position, particleHash);
			COMPUTE_CHECK("Kernel execution failed: CalculateHashKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(positionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void Reorder(uint16_t oldPositionVBO, glm::vec3* oldVelocity, glm::vec3* sortedPosition, glm::vec3* sortedVelocity,
			glm::uvec2* particleHash, uint32_t* cellStart, uint32_t particleCount, uint32_t cellCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			// Set all indices of the array to '0xffffffff'
			COMPUTE_SAFE(cudaMemset(cellStart, 0xffffffff, cellCount * sizeof(uint32_t)));

			// Buffer data
			glm::vec3* oldPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));

			// Kernel
			ReorderKernel << < blockCount, threadCount >> > (particleHash, cellStart, oldPosition, oldVelocity, sortedPosition, sortedVelocity);
			COMPUTE_CHECK("Kernel execution failed: ReorderKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(oldPositionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void Collide(uint16_t positionVBO, glm::vec3* sortedPosition, glm::vec3* sortedVelocity,
			glm::vec3* oldVelocity, glm::vec3* newVelocity, float* pressure, float* density,
			glm::uvec2* particleHash, uint32_t* cellStart, uint32_t particleCount, uint32_t cellCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 64, blockCount, threadCount);

			// Buffer data
			glm::vec3* newPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&newPosition, positionVBO));

			// Kernel
			CalculateDensityKernel <<< blockCount, threadCount >>> (sortedPosition, pressure, density, particleHash, cellStart);
			COMPUTE_CHECK("Kernel execution failed: CalculateDensityKernel");

			COMPUTE_SAFE(cudaDeviceSynchronize());

			// Kernel
			CalculateForceKernel <<< blockCount, threadCount >>> (newPosition, newVelocity, sortedPosition, sortedVelocity, pressure, density, particleHash, cellStart);
			COMPUTE_CHECK("Kernel execution failed: CalculateForceKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(positionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}
	}
}