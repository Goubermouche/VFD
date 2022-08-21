#ifndef FLIP_SIMULATION_H
#define FLIP_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "SimulationData.cuh"

namespace fe {
	struct FLIPSimulationDescription {
		float TimeStep;

		uint32_t SubStepCount;
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
	public:
		bool paused = false;
	private:
		FLIPSimulationDescription m_Description;
		flip::SimulationData m_Data;

		Ref<VertexBuffer> m_PositionVBO[2];
		Ref<VertexArray> m_PositionVAO[2];

		uint32_t m_CurrentPositionRead;
		uint32_t m_CurrentPositionWrite;

		bool m_Initialized = false;
	};
}

#endif // !FLIP_SIMULATION_H