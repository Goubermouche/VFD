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

		void FLIPUploadMACVelocities(MACVelocityField& data)
		{
			COMPUTE_SAFE(cudaMemcpyToSymbol(c_MAC, &data, sizeof(data))) // !
			printf("[MAC]   velocities uploaded [%d bytes]", sizeof(data));
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