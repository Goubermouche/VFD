#include "pch.h"
#include "Simulation.cuh"

#include "Simulation/FLIP/SimulationKernel.cu"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	namespace flip {
		extern "C" {
			void UploadSimulationData(SimulationData& data)
			{
				COMPUTE_SAFE(cudaMemcpyToSymbol(c_Description, &data, sizeof(SimulationData)))
			}
		}
	}
}