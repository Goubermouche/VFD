#include "pch.h"

#include "Compute/Utility/RadixSort/RadixSort.cuh"
#include "Simulation/SPH/SimulationKernel.cu"
#include "Compute/Utility/CudaKernelUtility.cuh"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	extern "C" {
		void SetParameters(SimulationData& params) {
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_Description, &params, sizeof(SimulationData)));
		}

		void Integrate(unsigned int oldPositionVBO, unsigned int newPositionVBO, glm::vec4* oldVelocity, glm::vec4* newVelocity, int particleCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			// Buffer data
			glm::vec4* oldPosition;
			glm::vec4* newPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&newPosition, newPositionVBO));

			// Kernel
			IntegrateKernel << < blockCount, threadCount >> > (newPosition, oldPosition, newVelocity, oldVelocity);
			COMPUTE_CHECK("Kernel execution failed: IntegrateKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(oldPositionVBO));
			COMPUTE_SAFE(cudaGLUnmapBufferObject(newPositionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void CalculateHash(unsigned int positionVBO, glm::uvec2* particleHash, int particleCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 512, blockCount, threadCount);

			// Buffer data
			glm::vec4* position;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&position, positionVBO));

			// Kernel
			CalculateHashKernel << < blockCount, threadCount >> > (position, particleHash);
			COMPUTE_CHECK("Kernel execution failed: CalculateHashKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(positionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void Reorder(unsigned int oldPositionVBO, glm::vec4* oldVelocity, glm::vec4* sortedPosition, glm::vec4* sortedVelocity,
			glm::uvec2* particleHash, unsigned int* cellStart, unsigned int particleCount, unsigned int cellCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			// Set all indices of the array to '0xffffffff'
			COMPUTE_SAFE(cudaMemset(cellStart, 0xffffffff, cellCount * sizeof(unsigned int)));

			// Buffer data
			glm::vec4* oldPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));

			// Texture data
			unsigned int float4MemorySize = particleCount * sizeof(glm::vec4);
			COMPUTE_SAFE(cudaBindTexture(0, oldPositionTexture, oldPosition, float4MemorySize));
			COMPUTE_SAFE(cudaBindTexture(0, oldVelocityTexture, oldVelocity, float4MemorySize));

			// Kernel
			ReorderKernel << < blockCount, threadCount >> > (particleHash, cellStart, oldPosition, oldVelocity, sortedPosition, sortedVelocity);
			COMPUTE_CHECK("Kernel execution failed: ReorderKernel");

			// Unbind textures
			COMPUTE_SAFE(cudaUnbindTexture(oldPositionTexture));
			COMPUTE_SAFE(cudaUnbindTexture(oldVelocityTexture));

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(oldPositionVBO));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}

		void Collide(unsigned int positionVBO, glm::vec4* sortedPosition, glm::vec4* sortedVelocity,
			glm::vec4* oldVelocity, glm::vec4* newVelocity, float* pressure, float* density,
			glm::uvec2* particleHash, unsigned int* cellStart, unsigned int particleCount, unsigned int cellCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 64, blockCount, threadCount);

			// Buffer data
			glm::vec4* newPosition;
			COMPUTE_SAFE(cudaGLMapBufferObject((void**)&newPosition, positionVBO));

			// Texture data
			unsigned int float4MemorySize = particleCount * sizeof(glm::vec4);
			unsigned int float1MemorySize = particleCount * sizeof(float);
			COMPUTE_SAFE(cudaBindTexture(0, oldPositionTexture, sortedPosition, float4MemorySize));
			COMPUTE_SAFE(cudaBindTexture(0, oldVelocityTexture, sortedVelocity, float4MemorySize));
			COMPUTE_SAFE(cudaBindTexture(0, pressureTexture, pressure, float1MemorySize));
			COMPUTE_SAFE(cudaBindTexture(0, densityTexture, density, float1MemorySize));
			COMPUTE_SAFE(cudaBindTexture(0, particleHashTexture, particleHash, particleCount * sizeof(glm::uvec2)));
			COMPUTE_SAFE(cudaBindTexture(0, cellStartTexture, cellStart, cellCount * sizeof(unsigned int)));

			// Kernel
			CalculateDensityKernel << < blockCount, threadCount >> > (sortedPosition, pressure, density, particleHash, cellStart);
			COMPUTE_CHECK("Kernel execution failed: CalculateDensityKernel");

			COMPUTE_SAFE(cudaDeviceSynchronize());

			// Kernel
			CalculateForceKernel << < blockCount, threadCount >> > (newPosition, newVelocity, sortedPosition, sortedVelocity, pressure, density, particleHash, cellStart);
			COMPUTE_CHECK("Kernel execution failed: CalculateForceKernel");

			// Unbind buffers
			COMPUTE_SAFE(cudaGLUnmapBufferObject(positionVBO));

			// Unbind textures
			COMPUTE_SAFE(cudaUnbindTexture(oldPositionTexture));
			COMPUTE_SAFE(cudaUnbindTexture(oldVelocityTexture));
			COMPUTE_SAFE(cudaUnbindTexture(pressureTexture));
			COMPUTE_SAFE(cudaUnbindTexture(densityTexture));
			COMPUTE_SAFE(cudaUnbindTexture(particleHashTexture));
			COMPUTE_SAFE(cudaUnbindTexture(cellStartTexture));

			COMPUTE_SAFE(cudaDeviceSynchronize());
		}
	}
}