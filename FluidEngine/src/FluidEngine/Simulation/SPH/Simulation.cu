#include "FluidEngine/Compute/Utility/CUDA/cutil.h"
#include "FluidEngine/Compute/Utility/RadixSort/RadixSort.cuh"
#include "FluidEngine/Simulation/SPH/SimulationKernel.cu"
#include "FluidEngine/Compute/Utility/CudaKernelUtility.cuh"

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace fe {
	extern "C" {
		void SetParameters(SimulationData& params) {
			printf("parameters set!\n");
			CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Description, &params, sizeof(SimulationData)));
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
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&newPosition, newPositionVBO));

			// Kernel
			IntegrateKernel <<< blockCount, threadCount >>> (newPosition, oldPosition, newVelocity, oldVelocity);
			CUT_CHECK_ERROR("Kernel execution failed: IntegrateKernel");

			// Unbind buffers
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(oldPositionVBO));
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(newPositionVBO));

			CUDA_SAFE_CALL(cudaDeviceSynchronize())
		}

		void CalculateHash(unsigned int positionVBO, glm::uvec2* particleHash, int particleCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 512, blockCount, threadCount);

			// Buffer data
			glm::vec4* position;
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&position, positionVBO));

			// Kernel
			CalculateHashKernel <<< blockCount, threadCount >>> (position, particleHash);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateHashKernel");

			// Unbind buffers
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(positionVBO));

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}

		void Reorder(unsigned int oldPositionVBO, glm::vec4* oldVelocity, glm::vec4* sortedPosition, glm::vec4* sortedVelocity,
			glm::uvec2* particleHash, unsigned int* cellStart, unsigned int particleCount, unsigned int cellCount)
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(particleCount, 256, blockCount, threadCount);

			// Set all indices of the array to '0xffffffff'
			CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, cellCount * sizeof(unsigned int)));

			// Buffer data
			glm::vec4* oldPosition;
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPosition, oldPositionVBO));

			// Texture data
			unsigned int float4MemorySize = particleCount * sizeof(glm::vec4);
			CUDA_SAFE_CALL(cudaBindTexture(0, oldPositionTexture, oldPosition, float4MemorySize));
			CUDA_SAFE_CALL(cudaBindTexture(0, oldVelocityTexture, oldVelocity, float4MemorySize));

			// Kernel
			ReorderKernel << < blockCount, threadCount >> > (particleHash, cellStart, oldPosition, oldVelocity, sortedPosition, sortedVelocity);
			CUT_CHECK_ERROR("Kernel execution failed: ReorderKernel");

			// Unbind textures
			CUDA_SAFE_CALL(cudaUnbindTexture(oldPositionTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(oldVelocityTexture));

			// Unbind buffers
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(oldPositionVBO));

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&newPosition, positionVBO));

			// Texture data
			unsigned int float4MemorySize = particleCount * sizeof(glm::vec4);
			unsigned int float1MemorySize = particleCount * sizeof(float);
			CUDA_SAFE_CALL(cudaBindTexture(0, oldPositionTexture, sortedPosition, float4MemorySize));
			CUDA_SAFE_CALL(cudaBindTexture(0, oldVelocityTexture, sortedVelocity, float4MemorySize));
			CUDA_SAFE_CALL(cudaBindTexture(0, pressureTexture, pressure, float1MemorySize));
			CUDA_SAFE_CALL(cudaBindTexture(0, densityTexture, density, float1MemorySize));
			CUDA_SAFE_CALL(cudaBindTexture(0, particleHashTexture, particleHash, particleCount * sizeof(glm::uvec2)));
			CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTexture, cellStart, cellCount * sizeof(unsigned int)));

			// Kernel
			CalculateDensityKernel <<< blockCount, threadCount >>> (sortedPosition, pressure, density, particleHash, cellStart);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateDensityKernel");

			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			// Kernel
			CalculateForceKernel <<< blockCount, threadCount >>> (newPosition, newVelocity, sortedPosition, sortedVelocity, pressure, density, particleHash, cellStart);
			CUT_CHECK_ERROR("Kernel execution failed: CalculateForceKernel");

			// Unbind buffers
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(positionVBO));

			// Unbind textures
			CUDA_SAFE_CALL(cudaUnbindTexture(oldPositionTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(oldVelocityTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(pressureTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(densityTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(particleHashTexture));
			CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTexture));

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}
	}
}