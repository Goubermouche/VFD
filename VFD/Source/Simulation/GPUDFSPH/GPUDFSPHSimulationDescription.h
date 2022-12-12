#ifndef GPU_DFSPH_SIMULATION_DESCRIPTION_H
#define GPU_DFSPH_SIMULATION_DESCRIPTION_H

namespace vfd
{
	struct GPUDFSPHSimulationDescription
	{
		// Time step
		float TimeStepSize = 0.001f;
		float MinTimeStepSize = 0.0001f;
		float MaxTimeStepSize = 0.005f;

		float ParticleRadius = 0.025f;
		glm::vec3 Gravity = { 0.0f, -9.81f, 0.0f };
	};
}

#endif // !GPU_DFSPH_SIMULATION_DESCRIPTION_H