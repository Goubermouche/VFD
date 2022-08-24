#include "FLIPSimulation.cuh"

#include "Simulation/FLIP/FLIPSimulationKernel.cu"
#include "Compute/Utility/CudaKernelUtility.cuh"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	extern "C" {
		void FLIPUploadSimulationData(FLIPSimulationData& data)
		{
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_FLIPDescription, &data, sizeof(data)))
		}

		void FLIPUploadMAC(MAC& data)
		{
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_MACTest, &data, sizeof(MAC))) // !
		}

		void FLIPUploadMACVelocities(MACVelocityField& data)
		{
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_MAC, &data, sizeof(data))) // !
			LOG("velocities uploaded", "FLIP][MAC", ConsoleColor::Cyan);
		}

		void FLIPUpdateFluidSDF()
		{
			// Grid size
			int threadCount;
			int blockCount;
			ComputeGridSize(1, 256, blockCount, threadCount);
			FLIPTestKernel << < blockCount, threadCount >> > ();
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}
	}
}