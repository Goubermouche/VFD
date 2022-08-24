#ifndef FLIP_SIMULATION_H
#define FLIP_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Simulation/FLIP/FLIPSimulationParameters.cuh"

namespace fe {
	// TODO: 
	// - proper struct deallocation
	struct FLIPSimulationDescription {
		float TimeStep;

		uint32_t SubStepCount;

		glm::ivec3 Size; // Grid size
	};

	class FLIPSimulation : public RefCounted
	{
	public:
		FLIPSimulation(const FLIPSimulationDescription& desc);
		~FLIPSimulation();

		void OnUpdate();

		const Ref<VertexArray>& GetVAO() {
			return m_PositionVAO[m_CurrentPositionRead];
		}
	private:
		void InitMemory();
		void FreeMemory();

		void InitBoundary();
	public:
		bool paused = false;
	private:
		FLIPSimulationDescription m_Description;
		FLIPSimulationParameters m_Parameters; // device-side data

		Ref<VertexBuffer> m_PositionVBO[2];
		Ref<VertexArray> m_PositionVAO[2];

		uint32_t m_CurrentPositionRead;
		uint32_t m_CurrentPositionWrite;

		bool m_Initialized = false;

		// MACVelocityField m_MACVelocity;
		MACVelocityField m_MACVelocity;
		MACVelocityField m_MACVelocityDevice;

		ParticleLevelSet m_LiquidSDF;
		WeightGrid m_WeightGrid;
		Array3D<float> m_Viscosity;

		ValidVelocityComponent m_ValidVelocities;
	};
}

#endif // !FLIP_SIMULATION_H